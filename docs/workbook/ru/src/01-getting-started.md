# Начало работы

## Сценарий

Вы никогда не запускали gigastt и хотите получить рабочую локальную
транскрибацию примерно за пять минут: установить бинарник, скачать модель
GigaAM v3, транскрибировать первый аудиофайл — на macOS, Linux или в Docker.
Эта глава покрывает весь путь; другие документы для этого не понадобятся.

## Предпосылки

- **Диск:** ~1,5 ГБ свободно (модель + инструменты). «Лёгкий» путь с
  `--prequantized` требует при установке всего ~250 МБ.
- **RAM:** ~800 МБ свободно при дефолтном `--pool-size 2` (~400 МБ на сессию).
- **Сеть** (если вы не идёте по рецепту для замкнутого контура): доступ к
  `huggingface.co` (полная модель) или `github.com` (предквантизованный бандл).
- **Аудиофайл для транскрибации** — WAV, M4A, MP3, OGG или FLAC. Подойдёт
  любая короткая запись русской речи.
- Только для `cargo install` (сборка из исходников): Rust 1.88+ и `protoc` в
  `PATH` (`brew install protobuf` / `apt install protobuf-compiler`).

Выберите **один** рецепт ниже — macOS, Linux, Docker или замкнутый контур, —
затем один раз прочитайте [Выбор головы распознавания](#выбор-головы-распознавания)
и [Дорогой первый запуск](#дорогой-первый-запуск).

## Рецепт: macOS (Homebrew)

Homebrew — самый быстрый путь на Apple Silicon (tap содержит бинарник с
CoreML). На Intel Mac используйте `cargo install gigastt` — см. требование
про protoc в рецепте для Linux.

```sh
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt

# Скачать модель (~850 МБ FP32 с HuggingFace, затем разовая ~2-минутная
# INT8-квантизация — см. «Дорогой первый запуск» ниже):
gigastt download

# Транскрибировать первый файл:
gigastt transcribe recording.wav
```

**Проверка:** последняя команда печатает распознанный текст в stdout, например:

```text
$ gigastt transcribe recording.wav
Привет, как дела?
```

а `ls ~/.gigastt/models/` показывает `v3_rnnt_encoder_int8.onnx`,
`v3_rnnt_decoder.onnx`, `v3_rnnt_joint.onnx` и `v3_vocab.txt`.

## Рецепт: Linux (готовый бинарник или cargo)

**Вариант A — готовый бинарник** (без Rust-инструментария и protoc). Каждый
релиз публикует tarball'ы для `x86_64-unknown-linux-gnu` и
`aarch64-unknown-linux-gnu`:

```sh
# Определить тег последнего релиза (или задайте TAG=v2.13.0 вручную):
TAG=$(curl -fsSL https://api.github.com/repos/ekhodzitsky/gigastt/releases/latest \
      | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
VER=${TAG#v}

curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz"
curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz.sha256"
sha256sum -c "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz.sha256"

tar xf "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz"
sudo install -m 0755 gigastt /usr/local/bin/gigastt
```

(На ARM64 замените `x86_64-unknown-linux-gnu` на
`aarch64-unknown-linux-gnu`. Homebrew на Linux x86_64 — `brew install
gigastt` после tap из рецепта для macOS — тоже работает.)

**Вариант B — cargo** (любая платформа, нужны Rust 1.88+ и `protoc`):

```sh
sudo apt install protobuf-compiler   # Debian/Ubuntu; пропустите, если protoc есть
cargo install gigastt
```

Затем скачайте модель «лёгким» способом — предквантизованный INT8-бандл
~225 МБ из закреплённого GitHub Release (без ~850 МБ FP32-загрузки и без
~2-минутной квантизации на устройстве; удобно и тогда, когда HuggingFace
недоступен, а GitHub — нет):

```sh
gigastt download --prequantized
gigastt transcribe recording.wav
```

**Проверка:** `gigastt transcribe recording.wav` печатает распознанный текст в
stdout, а `ls ~/.gigastt/models/` показывает файлы модели `v3_rnnt_*`.

## Рецепт: Docker

Готовые мультиархитектурные образы (amd64 + arm64) публикуются в GHCR для
каждого релиза; теги `-cuda` содержат CUDA-вариант:

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:latest   # в продакшене зафиксируйте :<version>

docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:latest
```

Именованный volume сохраняет модель между перезапусками контейнера; без него
контейнер будет заново скачивать ~850 МБ при каждом пересоздании. При первом
старте контейнер скачивает модель и квантизует её — порт поднимается сразу,
но инференс доступен, только когда `/ready` становится «зелёным»:

```sh
# Дождаться загрузки модели (503, пока идёт инициализация):
until curl -sf http://127.0.0.1:9876/ready > /dev/null; do sleep 5; done

curl http://127.0.0.1:9876/health
```

Затем транскрибируйте файл с хоста (путь к файлу — на хосте: его читает
`curl`, а не контейнер):

```sh
curl -F file=@recording.wav http://127.0.0.1:9876/v1/transcribe
```

**Проверка:** `/health` возвращает

```json
{"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.13.0","punctuation":true,"itn":true}
```

(поле `version` отражает скачанный образ), а POST возвращает JSON с
транскриптом:

```json
{"text":"Привет, как дела?","words":[{"word":"привет","start":0.0,"end":0.4,"confidence":0.99}],"duration":1.2}
```

## Рецепт: замкнутый контур (offline bundle)

Для машин без доступа в интернет каждый релиз публикует самодостаточный
офлайн-бандл для каждой Linux-цели — бинарник + предквантизованная INT8-модель
`rnnt` + модель пунктуации + systemd unit + установщик, — а также два
Debian-пакета с тем же содержимым. Скачайте их на **подключённой** машине,
перенесите и установите.

Поток с tarball'ом (любой дистрибутив):

```sh
# На подключённой машине (TAG/VER — см. рецепт для Linux):
curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz"
curl -fLO "https://github.com/ekhodzitsky/gigastt/releases/download/${TAG}/gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256"
sha256sum -c "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256"

# На целевой машине:
tar xf "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz"
cd "gigastt-${VER}-offline-x86_64-unknown-linux-gnu"
sudo ./install.sh                      # проверяет SHA256SUMS, ставит бинарник + модель + unit
sudo systemctl enable --now gigastt
```

Поток с Debian-пакетами: установите `gigastt_<ver>_amd64.deb` (бинарник +
unit) вместе с `gigastt-model-int8_<ver>_all.deb` (тот же набор моделей),
затем `sudo systemctl enable --now gigastt`.

В бандл намеренно не входят опциональные части — диаризация спикеров и головы
`e2e_rnnt` / `ml_ctc`. Установленный unit работает с `GIGASTT_OFFLINE=1`,
поэтому отсутствующая опциональная модель — это быстрая, понятная ошибка с
точным путём, куда положить файл (скачайте его на подключённой машине командой
`gigastt download` и скопируйте), а не сетевой таймаут. Полный список
содержимого и проверка подписей — в
[packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md).

**Проверка:** `curl http://127.0.0.1:9876/health` возвращает
`{"status":"ok",...}` с `"model":"gigaam-v3-rnnt"`, а
`gigastt transcribe sample.wav --model-dir /usr/share/gigastt/models` печатает
текст (флаг нужен только при запуске без установки — systemd unit уже
указывает на установленную модель).

## Выбор головы распознавания

gigastt поставляется с четырьмя головами распознавания; `--model-variant`
выбирает одну из них при `download` / `serve` / `transcribe`. Если флаг не
указан, существующая директория модели используется как есть
(автоопределение), а свежая установка по умолчанию получает `rnnt`.

| Голова | Языки | Стиль вывода | Когда выбирать |
|---|---|---|---|
| `rnnt` (по умолчанию) | русский | «Голый» lowercase из акустической модели; регистр + пунктуация восстанавливаются автоскачиваемым проходом RuPunct, цифры — ITN | По умолчанию: минимальный WER на русской речи |
| `e2e_rnnt` | русский | Пунктуация / регистр / ITN «зашиты» в акустическую модель | Нужна одна самодостаточная модель без постобработки |
| `ml_ctc` | ru/en/kk/ky/uz | «Голый» lowercase, без восстановления | Смешанная русско-английская (или kk/ky/uz) речь; лёгкий энкодер 220M |
| `ml_ctc_large` | ru/en/kk/ky/uz | «Голый» lowercase, без восстановления | Мультиязычная речь, где точность важнее размера (энкодер 600M) |

Головы `ml_ctc*` скачиваются сразу в предквантизованном INT8, поэтому шага
квантизации у них нет. Смена головы после установки:

```sh
gigastt download --model-variant e2e_rnnt   # скачать другую голову
gigastt serve --model-variant e2e_rnnt      # и явно подать её
```

Цифры WER/RTF по каждой голове — в
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md);
более глубокий обзор моделей и бэкендов — в главе
[Модели и бэкенды](04-models-and-backends.md).

## Дорогой первый запуск

Самый первый `gigastt download` (или первый `gigastt serve`, который
автоматически скачивает отсутствующую модель) делает две разовые вещи:

1. **Скачивает ~850 МБ** FP32 ONNX-файлов с HuggingFace (с проверкой SHA-256,
  через промежуточный `.partial` и атомарное переименование).
2. **Квантизует энкодер в INT8** (~2 минуты, один раз), создавая энкодер
   ~225 МБ, который движок реально загружает. Следующие запуски используют его
   повторно.

Три рычага меняют цену:

- `gigastt download --prequantized` — рекомендуемый короткий путь: скачать
  предквантизованный INT8-бандл ~225 МБ из закреплённого GitHub Release. Без
  FP32-загрузки, без локальной квантизации, без `protoc`. Обратите внимание:
  файлы берутся с `github.com`, а не с `huggingface.co` — полезно, когда один
  из двух хостов заблокирован.
- `gigastt download --skip-quantize` (или `GIGASTT_SKIP_QUANTIZE=1` у
  `serve`) — оставить FP32-энкодер и пропустить квантизацию. Движок тогда
  загружает FP32: инференс медленнее, а модель занимает в ~4 раза больше RAM.
  Только для отладки.
- Ничего — просто позволить первому `serve` сделать всё самому. Порт
  поднимается сразу; `/health` отвечает `200` с `"model":"loading"`, а
  `/ready` возвращает `503 {"reason":"initializing"}`, пока модель не готова,
  поэтому клиенты должны ждать `/ready`, а не сам факт запущенного процесса.

## Проверка результата

Сквозной чек-лист, работающий после любого из рецептов выше:

```sh
# 1. Файлы модели на месте:
ls ~/.gigastt/models/
#   v3_rnnt_encoder_int8.onnx  v3_rnnt_decoder.onnx  v3_rnnt_joint.onnx  v3_vocab.txt  ...

# 2. Офлайн-транскрибация работает (сервер не нужен):
gigastt transcribe recording.wav
#   → печатает распознанный текст в stdout

# 3. Сервер поднимается и сообщает загруженную голову:
gigastt serve &                      # Ctrl-C для остановки; по умолчанию http://127.0.0.1:9876
curl http://127.0.0.1:9876/ready     # 200, когда модель загружена
curl http://127.0.0.1:9876/health
#   {"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"...","punctuation":true,"itn":true}

# 4. REST-транскрибация работает:
curl -F file=@recording.wav http://127.0.0.1:9876/v1/transcribe
#   → {"text":"...","words":[...],"duration":N}
```

## Частые ошибки

- **`protoc` не найден** при `cargo install` или сборке из исходников —
  установите компилятор Protocol Buffers (`brew install protobuf` /
  `apt install protobuf-compiler`) или вовсе обойдитесь без инструментария,
  взяв готовый бинарник / Homebrew.
- **Первый `serve` «висит» несколько минут** — это разовая загрузка модели +
  INT8-квантизация, а не зависание: `/health` в это время возвращает
  `{"model":"loading"}`. Подготовьте модель заранее командой `gigastt download
  --prequantized` и настройте клиентов на ожидание `/ready`.
- **`Address already in use` на порту 9876** — найдите, кто держит порт:
  `lsof -nP -tiTCP:9876 -sTCP:LISTEN`; убедитесь, что это gigastt
  (`ps -p <pid> -o command=`), затем `kill <pid>` (SIGTERM корректно завершает
  сессии) или запустите на другом порту с `--port`.
- **Скачивание модели падает или виснет** (прокси, файрвол, HuggingFace
  недоступен) — повторите `gigastt download`; промежуточный `.partial`-файл
  делает повтор безопасным, а коды выхода различают причины (65 = контрольная
  сумма, 69 = сеть, 74 = диск). Если `huggingface.co` заблокирован, а
  `github.com` — нет, используйте `gigastt download --prequantized`; в полностью
  замкнутом контуре — офлайн-бандл. При ошибках диска проверьте права на
  `~/.gigastt/models/`.
- **OOM или активный swap при старте** — каждая сессия пула загружает свою
  копию энкодера (~400 МБ резидентно с INT8); дефолтный `--pool-size 2`
  достигает ~790 МБ. На слабых машинах запускайте с `--pool-size 1`.

Полная таблица «симптом → причина → исправление» — в
[docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md).

## Ссылки

- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) —
  канонический справочник CLI (все флаги и переменные окружения)
- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) —
  справочник REST / SSE / WebSocket API
- [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md) —
  цифры WER / RTF по каждой голове
- [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) —
  симптом → причина → исправление
- [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md) —
  детали Docker, reverse proxy, systemd, офлайн-установка
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) —
  контрольные суммы, minisign, SLSA provenance для артефактов релизов
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) —
  состав офлайн-бандла и опции установщика
- [Транскрибация файлов](02-file-transcription.md) — следующая глава: пакетная
  обработка, режим watch, форматы экспорта
- [Модели и бэкенды](04-models-and-backends.md) — головы, квантизация,
  execution providers в деталях
