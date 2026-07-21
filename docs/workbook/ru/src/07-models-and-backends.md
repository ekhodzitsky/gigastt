# Модели и бэкенды

## Сценарий

gigastt уже работает с головой `rnnt` по умолчанию на CPU-бэкенде, и теперь
нужно осознанно что-то поменять: другую голову распознавания (пунктуация «из
коробки» или языки помимо русского), более лёгкую загрузку модели, более
быстрый execution provider под ваше железо или больший пул сессий. Эта глава
отвечает на четыре вопроса проверяемыми рецептами: **какую голову**, **INT8
или FP32**, **какой бэкенд** и **сколько RAM** нужно пулу.

Цифры WER и RTF здесь **не дублируются** — канонические таблицы с доверительными
интервалами живут в
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md);
глава лишь ссылается на них. Флаги сверены с `gigastt <command> --help`; полный
справочник флагов —
[docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md).

## Предпосылки

- Установленный gigastt (бинарь, пакет или образ) — см.
  [Начало работы](01-getting-started.md).
- Диск: ~1,1 ГБ свободно для стандартного пути с FP32-загрузкой (FP32-набор
  плюс сгенерированный INT8-энкодер) или ~250 МБ для лёгкого pre-quantized пути.
- Для сборки нестандартного бэкенда из исходников: Rust 1.88+ и `protoc` в
  `PATH` (требования сборки:
  [docs/architecture.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/architecture.md)).

## Рецепт

### Выбор головы распознавания

Голова выбирается флагом `--model-variant` (env `GIGASTT_MODEL_VARIANT`) у
команд `serve` / `download` / `transcribe`. Все головы используют общий
mel-фронтенд и контракт входа 16 кГц моно; различаются ONNX-файлами,
словарём и декодированием.

| Голова | Размер на диске | Языки | Текст на выходе | Точность | Когда брать |
|---|---|---|---|---|---|
| `rnnt` (по умолчанию) | энкодер 844 МБ FP32 → ~215 МБ INT8 (авто-квантизация) + decoder/joiner/vocab (несколько МБ) | русский | «Голый» lowercase; дополняйте `--punctuation` / `--itn` (включены по умолчанию в режиме `auto`) | Лучший русский WER из четырёх — [таблица](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#accuracy-by-domain--wer--95-ci) | Русскоязычные нагрузки; дефолт не случаен |
| `e2e_rnnt` | Тот же класс размера, что у `rnnt` (~850 МБ FP32 → INT8 генерируется локально) | русский | Пунктуация / регистр / ITN **встроены**, один проход | WER выше, чем у `rnnt`, но лучший F1 пунктуации/регистра — [сравнение](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#punctuation-quality--e2e_rnnt-vs-rnnt--rupunct-restore) | Нужен читаемый русский текст за один проход, без шага восстановления |
| `ml_ctc` | ~225 МБ pre-quantized INT8, только энкодер (без decoder/joiner) | ru/en/kk/ky/uz | «Голый» lowercase | [Мультиязычные таблицы](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#english--wer--librispeech-test-clean) | Мультиязычное аудио при малом футпринте |
| `ml_ctc_large` | ~592 МБ pre-quantized INT8, только энкодер | ru/en/kk/ky/uz | «Голый» lowercase | Лучшая мультиязычная точность; на чистом русском чтении приближается к `rnnt` — [таблица](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#accuracy-by-domain--wer--95-ci) | Смешанные языки или английский/казахский/кыргызский/узбекский как таковые |

Два жёстких ограничения, следующих из таблицы:

- У `rnnt` / `e2e_rnnt` **кириллический** словарь — английский они не могут
  транскрибировать в принципе. Для английского (или kk/ky/uz) подходят только
  Multilingual-головы.
- Multilingual-головы — это encoder-only CTC: они всегда поставляются и
  работают как pre-quantized INT8 — ни FP32-загрузки, ни шага квантизации на
  устройстве для них не существует.

Загрузка и запуск другой головы:

```sh
gigastt download --model-variant e2e_rnnt
gigastt serve --model-variant e2e_rnnt
```

Правила автоопределения (из `crates/gigastt-core/src/model/mod.rs`):

- Без `--model-variant` движок определяет установленную голову по файлам в
  каталоге модели и использует полный комплект **как есть** (вообще без
  сетевых запросов).
- Явный `--model-variant`, отличный от установленной головы, → запрошенный
  набор скачивается **рядом**; головы никогда не смешиваются в одном инференсе,
  в лог пишется предупреждение. Если файлы нескольких голов сосуществуют, а
  вариант не задан, автоопределение предпочитает `rnnt`.
- Пустой каталог модели + отсутствие флага → скачивается дефолтный `rnnt`.

**Проверка:**

```sh
curl -s http://127.0.0.1:9876/health
# {"status":"ok","model":"gigaam-v3-e2e-rnnt","variant":"e2e_rnnt",...}
curl -s http://127.0.0.1:9876/v1/models
# .id / .name отражают загруженную голову; .encoder показывает int8 или fp32
```

### INT8 или FP32

Короткий ответ: **всегда INT8, если только вы не отлаживаете саму модель.**
INT8-энкодер работает как настоящие целочисленные вычисления (ядра
`DynamicQuantizeLinear` + `MatMulInteger`/`ConvInteger`), сжимает энкодер
844 МБ → 215 МБ (~3,9×) и даёт ~0% деградации WER — цифры и методология:
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md#headline-single-engine-metrics).

Что происходит на каждом пути (RNN-T-головы):

- **По умолчанию** — `gigastt download` (или первый `serve` / `transcribe`)
  скачивает FP32-набор с HuggingFace (`istupakov/gigaam-v3-onnx`), затем
  однократно (~2 мин) выполняет квантизацию нативным Rust-кодом и пишет
  `v3_rnnt_encoder_int8.onnx` рядом. Движок **предпочитает INT8-энкодер**,
  когда файл присутствует.
- **Лёгкий путь** — `gigastt download --prequantized` скачивает готовый
  INT8-бандл (INT8-энкодер + decoder + joiner + vocab) из закреплённого
  GitHub Release: ни ~844 МБ FP32-загрузки, ни ~2-минутной квантизации. Это
  также запасной вариант, когда HuggingFace недоступен, а GitHub — нет.
- **Multilingual** — `ml_ctc` / `ml_ctc_large` скачивают pre-quantized
  INT8-энкодер istupakov'а напрямую с HuggingFace; `--prequantized` для них —
  холостое уточнение (отдельного бандла нет).
- **Вручную** — `gigastt quantize [--force]` повторно запускает квантизацию
  для головы, определённой в `--model-dir` (например, после подмены FP32-
  энкодера своим файнтюном).
- **Отказ** — `--skip-quantize` (env `GIGASTT_SKIP_QUANTIZE=1`) пропускает шаг
  квантизации; движок тогда загружает FP32-энкодер с ~4× расходом RAM на слот
  пула и более медленными CPU-ядрами. Оставьте это только для отладки.

**Проверка:**

```sh
ls ~/.gigastt/models/
# v3_rnnt_encoder_int8.onnx присутствует рядом с decoder/joint/vocab
gigastt transcribe sample.wav 2>&1 | grep 'transcribe complete'
# ... encoder=int8/cpu ... rtf=0.1xx
```

Поле `encoder=int8/<backend>` в строке лога о завершении — истина о том, какой
файл энкодера был загружен.

### Бэкенд под ваше железо

Бэкенд — это **compile-time Cargo-фича**, а не runtime-флаг: провайдер зашит в
бинарь, который вы устанавливаете или собираете:

| Ваше железо | Фича | Провайдер | Примечания |
|---|---|---|---|
| Любое (по умолчанию) | — | CPU (ONNX Runtime) | Эталонная сборка; RTF заметно ниже 1.0 с INT8-энкодером |
| macOS ARM64 (M1–M4) | `--features coreml` | CoreML + Neural Engine | Готовые релизные бинари macOS **уже собраны с этой фичей**; ~3× энкодер на коротких клипах, ~5,6× на длинных файлах против CPU |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ | Готового tarball нет — берите Docker-образ `-cuda` или `Dockerfile.cuda`; при отсутствии GPU в рантайме откатывается на CPU |
| Android / ARM64 | `--features nnapi` | NNAPI (NPU/DSP) | Не является взаимоисключающей с остальными |
| macOS ARM64, экспериментально | `--features ane` | Нативный Apple Neural Engine (Core ML `.mlpackage`) | Только голова `rnnt`, ускорение файлового режима; см. ниже |
| Apple Silicon, экспериментально | `--features candle` | Чистый Rust Candle на Metal | Только голова `rnnt`, FP32; см. ниже |

Соберите подходящий вариант:

```sh
cargo build --release                      # CPU, любая платформа
cargo build --release --features coreml    # macOS ARM64
cargo build --release --features cuda      # Linux x86_64 + NVIDIA (CUDA 12+)
cargo build --release --features nnapi     # Android / ARM64
```

Исключительность проверяется на этапе компиляции: `coreml` и `cuda` взаимно
исключают друг друга; `ane` конфликтует с `coreml`/`cuda`/`nnapi`/`candle`;
`candle` конфликтует с `coreml`/`cuda`. На плохой комбинации срабатывает
`compile_error!`.

Поведение в рантайме, о котором стоит знать:

- **Откат на CPU — намеренный, никогда не падение.** CoreML-сборка при старте
  выполняет ~1-секундную warmup-проверку; при неудаче пишет в лог `falling
  back to CPU execution provider` и пересобирает сессии на CPU. CUDA-бинарь
  аналогично работает на CPU, когда GPU не виден (например, контейнер запущен
  без `--gpus all`).
- **Упаковка CUDA.** Готового CUDA-tarball нет — релизная матрица собирает
  CPU-бинари для Linux (x86_64, aarch64), Windows и CoreML-сборку для macOS
  ARM64. Для GPU берите опубликованный образ
  `ghcr.io/ekhodzitsky/gigastt:<ver>-cuda` (рецепты:
  [Развёртывание и эксплуатация](06-deployment-ops.md)) или собирайте через
  [Dockerfile.cuda](https://github.com/ekhodzitsky/gigastt/blob/main/Dockerfile.cuda).
- **Нативный ANE-бэкенд** (`--features ane`, macOS ARM64): выполняет энкодер
  `rnnt` на Neural Engine через пакеты Core ML с фиксированными формами по
  бакетам, ~10× тёплого end-to-end ускорения против CPU-сборки (упор в
  декодирование). Пакеты скачиваются командой `gigastt download --ane` (в
  `~/.gigastt/models/ane/`); модель `e2e_rnnt` прозрачно откатывается на
  `ort`-энкодер, а стриминговые окна всегда идут по CPU-пути — ANE является
  ускорителем файлового режима. Полный дизайн и честные цифры:
  [docs/ane-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/ane-backend.md).
- **Candle-бэкенд** (`--features candle`, экспериментально): чисто-Rust
  инференс на Metal GPU, побайтовая идентичность с `ort`, только голова
  `rnnt`, FP32-веса, однократно конвертируемые скриптом
  `scripts/convert_gigaam_candle.py`. Подробности:
  [docs/candle-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/candle-backend.md).

**Проверка:**

```sh
gigastt transcribe sample.wav 2>&1 | grep 'transcribe complete'
# encoder=int8/coreml  → активна CoreML-сборка (int8/cuda, int8/cpu, ...)
# encoder=int8/ane     → энкодер обработал нативный ANE-бэкенд
```

На CoreML-сборке также смотрите стартовый лог: отсутствие строки `falling back
to CPU execution provider` означает, что warmup-проверка прошла.

### Файлы модели в замкнутом контуре

Всё, что нужно движку, живёт в одном каталоге — `~/.gigastt/models/` по
умолчанию, переопределяется флагом `--model-dir` у `serve` / `download` /
`transcribe` / `quantize`. Поэтому набор моделей — это обычный файловый
артефакт, который можно подготовить и переносить:

```sh
# На машине с сетью:
gigastt download --prequantized --model-dir /srv/gigastt-models

# Любым способом скопируйте на целевой хост (rsync, USB, хранилище артефактов):
rsync -a /srv/gigastt-models/ offline-host:/srv/gigastt-models/

# На офлайн-хосте — запретить любые сетевые обращения, падать сразу, а не висеть:
gigastt --offline serve --model-dir /srv/gigastt-models
```

Свойства, которые делают это безопасным:

- **Целостность проверяет бинарь, а не вы.** Каждая загрузка проверяется по
  SHA-256 против контрольных сумм, закреплённых в коде, складывается в
  `.partial` и атомарно переименовывается на место; повреждённый файл
  удаляется, а не принимается. Несовпадение контрольной суммы завершает
  процесс с кодом `65` (сеть `69`, диск `74`, Ctrl-C `130`) — пригодно для
  скриптов.
- **Полный набор модели = ноль сети.** При наличии полного комплекта головы
  (энкодер — INT8 или FP32 — плюс decoder/joiner/vocab для RNN-T-голов или
  INT8-энкодер + vocab для Multilingual) запуск не выполняет ни одного
  сетевого запроса, даже без `--offline`. `--offline` / `GIGASTT_OFFLINE=1`
  превращает любую *недостающую* опциональную модель (пунктуация, VAD,
  диаризация) в немедленную ошибку с именем файла, который нужно предоставить.
- **Голова автоопределяется по файлам** — скопированному каталогу не нужен
  флаг `--model-variant` на целевом хосте.
- При копировании включайте `*_int8.onnx`-энкодер (или скопируйте также FP32-
  энкодер и выполните `gigastt quantize --model-dir …` на целевом хосте),
  vocab и — для `rnnt`/`e2e_rnnt` — decoder и joiner.

Для полностью упакованной офлайн-установки (tarball с бинарём + INT8-моделью +
моделью пунктуации + systemd-юнит или вариант из двух deb-пакетов) берите
релизный offline-бандл — рецепт и шаги проверки в
[Развёртывание и эксплуатация](06-deployment-ops.md); состав —
[README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md).

**Проверка:**

```sh
gigastt --offline transcribe sample.wav --model-dir /srv/gigastt-models
# транскрибирует без сети; недостающий файл — немедленная ошибка с именем файла
```

### Память под пул сессий

Каждый слот пула десериализует **свою копию энкодера**, поэтому RSS растёт
линейно с `--pool-size` (по умолчанию 2). Движок закладывает на слот примерно
`2 × размер-файла-энкодера` резидентной памяти (измерено ~1,9× на INT8-
энкодере `rnnt`, CPU-провайдер, release-сборка):

| Голова (как загружена) | Файл энкодера | ≈ RAM на слот пула | Дефолтный пул 2 |
|---|---|---|---|
| `rnnt` / `e2e_rnnt` INT8 | ~215 МБ | ~0,4 ГБ | ~790 МБ суммарного RSS |
| `rnnt` / `e2e_rnnt` FP32 (`--skip-quantize`) | 844 МБ | ~1,6 ГБ | ~3,3 ГБ — никогда в проде |
| `ml_ctc` INT8 | ~225 МБ | ~0,45 ГБ | ~0,9 ГБ |
| `ml_ctc_large` INT8 | ~592 МБ | ~1,2 ГБ | ~2,4 ГБ |

Встроены две защиты:

- **Авто-ограничение по RAM.** При загрузке запрошенный пул урезается так,
  чтобы энкодеры пула оставались в пределах половины общей RAM — при срабатывании
  пишется предупреждение `Capping pool size N -> M`. Ограничение никогда не
  повышает ваш запрос и никогда не опускается ниже 1.
- **Деградированный запуск.** `--pool-min-size 1` (по умолчанию) позволяет
  серверу стартовать на частично загруженном пуле вместо падения, когда память
  заканчивается посреди загрузки.

Эмпирическое правило: `RAM ≥ pool_size × расход-на-слот + ~1 ГБ на ОС и пики
запросов`. На машине с 4 ГБ это означает `--pool-size 1–2` с INT8-энкодером —
тот же вывод, что и в пункте про OOM в
[Развёртывание и эксплуатация](06-deployment-ops.md).

**Проверка:**

```sh
# нет предупреждения "Capping pool size" при старте, и:
curl -s http://127.0.0.1:9876/ready
# {"status":"ready","pool_available":2,"pool_total":2}
```

## Проверка результата

Сквозной смоук после любого изменения из этой главы:

```sh
ls ~/.gigastt/models/                  # полный набор файлов головы, включая *_int8.onnx + vocab
gigastt transcribe sample.wav 2>&1 | grep 'transcribe complete'
# encoder=<int8|fp32>/<cpu|coreml|cuda|ane|candle>, rtf заметно ниже 1.0
curl -s http://127.0.0.1:9876/health   # "model"/"variant" соответствуют выбранной голове
curl -s http://127.0.0.1:9876/ready    # ready, pool_available >= 1
```

Затем сверьте ожидания по точности с
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md),
а не с кустарными замерами — харнесс, манифесты и нормализация значат больше,
чем секундомер.

## Частые ошибки

- **Английское аудио превращается в мусор с головой по умолчанию.** Это не
  баг: у `rnnt` / `e2e_rnnt` кириллический словарь, и латиницу они выдавать не
  могут в принципе. Переключитесь на `--model-variant ml_ctc` (или
  `ml_ctc_large`).
- **`--model-variant` словно игнорируется после первой загрузки.** Движок
  автоопределяет голову по каталогу модели; явный вариант, отличный от
  установленного, инициирует *вторую* загрузку рядом (с предупреждением
  `variants are never mixed`), а следующий запуск без флага снова предпочтёт
  `rnnt`. Удалите файлы неиспользуемой головы, чтобы каталог был однозначным,
  и верните место на диске.
- **Первый `serve` выглядит зависшим.** Это одноразовая загрузка ~850 МБ FP32
  + ~2 мин квантизации; `/health` отвечает `200` с `model:"loading"`, пока
  `/ready` остаётся `503 initializing`. Уберите это окно командой `gigastt
  download --prequantized` и стробируйте клиентов по `/ready`, никогда по
  `/health`.
- **OOM после переключения на `ml_ctc_large`.** Каждый слот теперь стоит ~1,2
  ГБ. Снизьте `--pool-size`, держите `--pool-min-size 1`, чтобы тесный хост
  загружался деградированно, и следите за предупреждением `Capping pool size`
  при старте.
- **CoreML-сборка не быстрее CPU.** Ищите в стартовом логе `falling back to
  CPU execution provider` — warmup-проверка не прошла, и движок (намеренно)
  работает на CPU. Поле `encoder=int8/cpu` в логе завершения это подтверждает.
- **CUDA-контейнер работает на скорости CPU.** GPU не виден: контейнеру нужны
  NVIDIA Container Toolkit и `--gpus all`. Бинарь молча откатывается на CPU —
  проверяйте `encoder=int8/cuda` в логе завершения.
- **`error: ane and coreml are mutually exclusive` (или похожая) при сборке.**
  Бэкенд-фичи конфликтуют по дизайну; собирайте ровно одну из `coreml` /
  `cuda` / `ane` / `candle` (`nnapi` — исключение).
- **`SHA-256 mismatch` во время `gigastt download`.** Стейджинговая загрузка
  повреждена или подменена; она удаляется, а не принимается, CLI завершается с
  кодом `65`. Просто перезапустите команду — не переименовывайте `.partial`
  вручную на место.

## Ссылки

- [Начало работы](01-getting-started.md) — установка и первая транскрипция
- [CLI и пакетная обработка](02-cli-batch.md) — рецепт по пропускной способности и памяти для офлайн-CLI
- [Развёртывание и эксплуатация](06-deployment-ops.md) — offline-бандл, systemd, пункты ранбука про OOM
- [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md) — канонические таблицы WER / RTF / футпринта
- [docs/architecture.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/architecture.md) — пайплайн, провайдеры, внутренности квантизации
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — полный справочник флагов
- [docs/ane-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/ane-backend.md) — нативный бэкенд Apple Neural Engine
- [docs/candle-backend.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/candle-backend.md) — бэкенд Candle/Metal
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) — проверка релизных артефактов
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) — состав offline-бандла
