<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Распознавание русской речи на устройстве с WER 10.4%</strong></p>
  <p align="center">Локальный STT-сервер на базе GigaAM v3 — без облака, без API-ключей, полная приватность</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/d/gigastt.svg" alt="crates.io downloads"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange" alt="Changelog"></a>
  </p>
  <p align="center"><a href="README.md">English</a> | <b>Русский</b></p>
</p>

---

**gigastt** превращает любой компьютер в сервер распознавания русской речи в реальном времени. Один бинарник, одна команда, точность на уровне лучших решений — всё работает локально.

```sh
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt && gigastt serve
# WebSocket: ws://127.0.0.1:9876/v1/ws
# REST API:  http://127.0.0.1:9876/v1/transcribe
```

## Почему gigastt?

| | gigastt | whisper.cpp | faster-whisper | Vosk | sherpa-onnx | Облачные API |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Модель** | GigaAM v3 | Whisper large-v3 | Whisper large-v3 | Vosk models | разные | вендор |
| **WER (русский)** | **10.4%** | ~18% | ~18% | ~20%+ | зависит от модели | 5–10% |
| **Языки** | русский | 99 | 99 | 20+ | 10+ | 100+ |
| **Стриминг** | WebSocket в реальном времени | — | — | WebSocket + gRPC | WebSocket + TCP | по-разному |
| **Задержка (16с, M1)** | **~700мс** | ~4с | ~2с | ~3с | ~1.5с | сеть |
| **Приватность** | 100% локально | 100% локально | 100% локально | 100% локально | 100% локально | данные уходят наружу |
| **Установка** | `cargo install` | cmake + make | `pip install` | `pip install` | cmake или pip | API-ключ + биллинг |
| **Реализация** | Rust | C/C++ | Python/C++ | C++/Java | C++ | N/A |
| **Биндинги** | Rust, C FFI | C, Python, Go, JS… | Python | Python, Java, JS, Go… | C, Python, Java, Swift… | SDK |
| **INT8 квантизация** | авто, 0% потери WER | GGML quant | CTranslate2 quant | — | — | N/A |
| **Параллельные сессии** | настраиваемый пул | 1 | 1 | 1 | 1 | лимиты провайдера |
| **Стоимость** | бесплатно | бесплатно | бесплатно | бесплатно | бесплатно | от $0.006/мин |

> **Компромисс:** gigastt поддерживает только русский язык. Для мультиязычного распознавания подойдут whisper.cpp или sherpa-onnx. Если нужна лучшая точность на русском локально — gigastt единственный Rust-нативный вариант на базе GigaAM v3, текущего SOTA для русского ASR. Обучена на **700K+ часах** русской речи. WER измерен на 993 записях Golos (4991 слово).

## Кому подойдёт?

- **Голосовые ассистенты в реальном времени** — WebSocket-стриминг с задержкой менее секунды
- **Транскрипция колл-центров** — диаризация спикеров + REST batch-обработка
- **Офлайн обработка документов** — транскрипция записей совещаний без загрузки в облако
- **Приватные мобильные приложения** — встраивание через C-ABI FFI на Android с on-device инференсом
- **Исследования и ML-пайплайны** — автономная библиотека `gigastt-core` для Rust ML-стеков

## Возможности

- **Стриминг в реальном времени** — частичная транскрипция по WebSocket во время речи
- **REST API + SSE** — транскрипция файлов с мгновенным или потоковым ответом
- **Аппаратное ускорение** — CoreML + Neural Engine (macOS), CUDA 12+ (Linux), CPU везде
- **INT8 квантизация** — модель в 4 раза меньше, на 43% быстрее
- **Множество форматов** — WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC
- **Диаризация спикеров** — определение кто говорит (опциональная фича)
- **Автопунктуация** — модель GigaAM v3 выдаёт текст с пунктуацией и нормализацией
- **Автозагрузка** — модель скачивается с HuggingFace при первом запуске (~850 МБ)
- **Docker** — образы для CPU и CUDA с многоэтапной сборкой
- **Защищённость** — лимиты соединений, ограничения фреймов, таймауты, санитизация ошибок

## Быстрый старт

### Установка и запуск

```sh
# Homebrew (macOS ARM64 / Linux x86_64)
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt
gigastt serve

# Из crates.io (нужен `protoc`: `brew install protobuf` / `apt install protobuf-compiler`)
cargo install gigastt
gigastt serve

# Из исходников
git clone https://github.com/ekhodzitsky/gigastt
cd gigastt
cargo run --release -- serve
```

Модель (~850 МБ) скачивается автоматически при первом запуске.

### Docker

```sh
# CPU (любая платформа)
docker build -t gigastt .
docker run -p 9876:9876 gigastt

# CUDA (Linux, требуется NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Модель скачивается при первом запуске (~850 МБ)
```

#### Образ с моделью внутри (baked)

```sh
# Обычный образ (модель скачивается при первом запуске, ~850 МБ)
docker build -t gigastt .

# Образ с моделью (нет задержки при старте, ~1.1 ГБ)
docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .
```

### Транскрипция файла

```sh
# CLI
gigastt transcribe recording.wav

# REST API
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @recording.wav
# {"text":"Привет, как дела?","words":[],"duration":3.5}
```

## API

### WebSocket — стриминг в реальном времени

Подключение к `ws://127.0.0.1:9876/v1/ws`, отправка PCM16 аудио-фреймов, получение транскрипции в реальном времени.

```
Клиент                            Сервер
  |                                 |
  |-------- connect --------------> |
  |                                 |
  | <------- ready ----------------- |
  | {type:"ready", version:"1.0"}  |
  |                                 |
  |------- configure (опционально)-> |
  | {type:"configure",              |
  |  sample_rate:16000}             |
  |                                 |
  |-------- binary PCM16 --------> |
  |                                 |
  | <------- partial --------------- |
  | {type:"partial", text:"привет"} |
  |                                 |
  | <------- final ----------------- |
  | {type:"final",                  |
  |  text:"Привет, как дела?"}      |
```

**Поддерживаемые частоты дискретизации:** 8, 16, 24, 44.1, 48 кГц (по умолчанию 48 кГц, внутри ресемплируется в 16 кГц).

### REST API

| Эндпоинт | Метод | Описание |
|---|---|---|
| `/health` | GET | Проверка состояния (`{"status":"ok"}`) |
| `/ready` | GET | Проба готовности (200 когда пул движка инициализирован) |
| `/v1/models` | GET | Информация о модели (тип encoder, размер пула, возможности) |
| `/v1/transcribe` | POST | Транскрипция файла, полный JSON-ответ |
| `/v1/transcribe/stream` | POST | Транскрипция файла с SSE-стримингом |
| `/v1/ws` | GET | WebSocket-апгрейд для стриминга в реальном времени |
| `/metrics` | GET | Prometheus-метрики (включается `--metrics`) |

**Пример SSE-стриминга:**

```sh
curl -X POST http://127.0.0.1:9876/v1/transcribe/stream \
  -H "Content-Type: application/octet-stream" \
  --data-binary @recording.wav
# data: {"type":"partial","text":"привет как"}
# data: {"type":"partial","text":"привет как дела"}
# data: {"type":"final","text":"Привет, как дела?"}
```

Полная спецификация протокола: [`docs/asyncapi.yaml`](docs/asyncapi.yaml)

#### Коды ошибок

| HTTP | Код | Когда |
|---|---|---|
| 400 | `bad_request` | Неверный формат аудио или некорректный запрос |
| 413 | `payload_too_large` | Файл превышает `--body-limit-bytes` (по умолчанию 50 МиБ) |
| 429 | `rate_limit_exceeded` | Исчерпан per-IP token bucket; заголовок `Retry-After` включён |
| 503 | `pool_saturated` | Все сессии инференса заняты; `Retry-After: 30` |
| 503 | `pool_closed` | Сервер завершает работу, пул закрыт для новых запросов |

```json
// Пример: насыщение пула
HTTP/1.1 503 Service Unavailable
Retry-After: 30

{"code":"pool_saturated","message":"All inference sessions are busy"}
```

### Клиентские библиотеки

Готовые WebSocket-клиенты в [`examples/`](examples/):

#### Python
```sh
pip install websockets
python examples/python_client.py recording.wav
```

#### Bun (TypeScript)
```sh
bun examples/bun_client.ts recording.wav
```

#### Go
```sh
# go mod init gigastt-client && go get github.com/gorilla/websocket
go run examples/go_client.go recording.wav
```

#### Kotlin
```sh
# Зависимости — см. заголовок KotlinClient.kt (Gradle/Maven)
kotlinc examples/KotlinClient.kt -include-runtime -d client.jar
java -jar client.jar recording.wav
```

## Производительность

| Метрика | Значение |
|---|---|
| **WER (русский)** | 10.4% (993 записи Golos, 4991 слово) |
| **INT8 vs FP32** | 0% деградации WER (10.4% vs 10.5% на 993 записях) |
| **Задержка (16с аудио, M1)** | ~700 мс (encoder 667 мс + decode 31 мс) |
| **Память (RSS)** | ~560 МБ |
| **Размер модели** | 851 МБ (FP32) / 222 МБ (INT8) |
| **Параллельные сессии** | до 4 (настраивается через `--pool-size`) |

### Аппаратное ускорение

| Платформа | Флаг компиляции | Execution Provider |
|---|---|---|
| macOS ARM64 (M1-M4) | `--features coreml` | CoreML + Neural Engine |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ |
| Любая платформа | _(по умолчанию)_ | CPU |

```sh
cargo build --release --features coreml   # macOS: CoreML + Neural Engine
cargo build --release --features cuda     # Linux: NVIDIA CUDA 12+
cargo build --release                     # CPU (любая платформа)
```

Фичи компилируются статически и взаимоисключающие.

### INT8 квантизация

Квантизированный encoder: в 4 раза меньше, ~43% быстрее, 0% деградации WER (проверено на 993 записях Golos / 4991 слово). Автоматически определяется при запуске.

Начиная с v0.9.0 квантизация всегда компилируется и автоматически вызывается при первом `download` или `serve` — ни feature-флага, ни ручных шагов не нужно. Cargo-фича `quantize` оставлена как no-op для обратной совместимости.

```sh
# Автоматически (рекомендуется)
cargo install gigastt
gigastt serve           # скачивает модель + автоквантизация при первом запуске

# Отключить автоквантизацию (оставить только FP32)
gigastt serve --skip-quantize
# или: GIGASTT_SKIP_QUANTIZE=1 gigastt serve

# Ручная переквантизация
gigastt quantize                     # нативная квантизация на Rust
gigastt quantize --force             # переквантизировать даже при наличии INT8-модели
```

## Структура проекта

gigastt организован как Cargo workspace из 3 крейтов:

| Крейт | Тип | Назначение |
|---|---|---|
| [`gigastt-core`](crates/gigastt-core) | lib (rlib) | Движок инференса, загрузка модели, квантизация, протокольные типы |
| [`gigastt-ffi`](crates/gigastt-ffi) | lib (cdylib) | C-ABI FFI-слой для встраивания в Android / мобильные приложения |
| [`gigastt`](crates/gigastt) | bin | Серверный бинарник (axum HTTP/WS) + CLI |

`gigastt-core` не зависит от серверных библиотек — встраивайте инференс в любой Rust-проект через `gigastt-core = "2.0"`.

## Архитектура

```
                    Аудио-вход
                   (PCM16, разные частоты)
                        |
                        v
               +-----------------+
               | Мел-спектрограмма |  64 bin, FFT=320, hop=160
               +-----------------+
                        |
                        v
            +------------------------+
            |   Conformer Encoder    |  16 слоёв, 768-dim, 240M параметров
            |  (ONNX Runtime)        |  CoreML | CUDA | CPU
            +------------------------+
                        |
                        v
            +------------------------+
            | RNN-T Decoder + Joiner |  Stateful: h/c сохраняются
            |  (ONNX Runtime)        |  между стриминг-чанками
            +------------------------+
                        |
                        v
            +------------------------+
            |   BPE-токенайзер       |  1025 токенов
            |   + автопунктуация     |
            +------------------------+
                        |
                        v
                  Русский текст
```

## Android / FFI

gigastt можно встроить в Android-приложения через C-ABI FFI-слой (без HTTP-сервера, без JNI).

```sh
# Собрать libgigastt_ffi.so для Android (arm64)
cargo ndk -t arm64-v8a -o ./jniLibs build --release -p gigastt-ffi
```

| Функция | Назначение |
|---|---|
| `gigastt_engine_new(model_dir)` | Загрузить движок (pool_size = 4 по умолчанию) |
| `gigastt_engine_new_with_pool_size(model_dir, pool_size)` | Загрузить с кастомным RAM-лимитом |
| `gigastt_transcribe_file(engine, wav_path)` | Синхронная транскрипция файла |
| `gigastt_stream_new(engine)` | Начать стриминговую сессию |
| `gigastt_stream_process_chunk(...)` | Передать PCM16-аудио, получить JSON-сегменты |
| `gigastt_stream_flush(...)` | Завершить стрим |

Фича `nnapi` на `gigastt-ffi` включает `ort/nnapi` для NPU/DSP-ускорения на Android: `cargo ndk ... build -p gigastt-ffi --features nnapi`. Для мобильных устройств рекомендуется `pool_size = 1` (~350 МБ RAM).

Полное руководство по интеграции: [`ANDROID.md`](ANDROID.md)  
Kotlin-мост: [`ffi/android/GigasttBridge.kt`](ffi/android/GigasttBridge.kt)

## Справка по CLI

Ключевые флаги для самых распространённых команд. У каждого флага есть переменная окружения — полный справочник в [`docs/cli.md`](docs/cli.md).

```sh
# Запустить сервер
gigastt serve --port 9876 --bind-all --metrics

# Транскрибировать файл
gigastt transcribe recording.wav

# Переквантизировать encoder (нативный Rust, ~2 мин одноразово)
gigastt quantize --force
```

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--port` | 9876 | Порт |
| `--host` | 127.0.0.1 | Адрес привязки (по умолчанию только loopback) |
| `--bind-all` | — | Разрешить привязку к не-loopback адресам |
| `--pool-size` | 4 | Параллельные сессии инференса |
| `--metrics` | — | Включить Prometheus на `/metrics` |
| `--idle-timeout-secs` | 300 | Таймаут неактивного WebSocket-соединения |
| `--max-session-secs` | 3600 | Максимальная длительность сессии |
| `--rate-limit-per-minute` | 0 | Rate limit по IP (0 = выключен) |
| `--skip-quantize` | — | Пропустить INT8-квантизацию при первом запуске |

## Модель

[**GigaAM v3 e2e_rnnt**](https://huggingface.co/istupakov/gigaam-v3-onnx) от [SberDevices](https://github.com/salute-developers/GigaAM):

| Свойство | Значение |
|---|---|
| Архитектура | RNN-T (Conformer encoder + LSTM decoder + joiner) |
| Encoder | 16-слойный Conformer, 768-dim, 240M параметров |
| Данные обучения | 700K+ часов русской речи |
| Словарь | 1025 BPE-токенов |
| Вход | 16 кГц моно PCM16 |
| Квантизация | INT8 доступна (v0.2+) |
| Лицензия | MIT |
| Размер загрузки | ~850 МБ (encoder 844 МБ, decoder 4.4 МБ, joiner 2.6 МБ) |

## Требования

| | macOS ARM64 | Linux x86_64 |
|---|---|---|
| **ОС** | macOS 14+ (Sonoma) | Любой современный дистрибутив |
| **CPU** | Apple Silicon (M1-M4) | x86_64 |
| **GPU** | _(встроенный, через CoreML)_ | NVIDIA + CUDA 12+ (опционально) |
| **Диск** | ~1.5 ГБ | ~1.5 ГБ |
| **RAM** | ~560 МБ | ~560 МБ |
| **Rust** | 1.85+ | 1.85+ |

## Безопасность

- **Loopback-only по умолчанию.** Сервер откажется слушать любой адрес кроме
  `127.0.0.1` / `::1` / `localhost`, пока оператор явно не передал `--bind-all`
  (или не задал `GIGASTT_ALLOW_BIND_ANY=1`). Защита от случайного публичного
  экспонирования за reverse-прокси или забытым port-forward.
- **Cross-origin запросы отклоняются по умолчанию.** Страница на
  `https://evil.example.com` больше не может drive-by-подключиться к локальному
  WebSocket / REST API. Loopback-источники всегда разрешены; остальные — через
  `--allow-origin https://app.example.com` (повторяемый флаг). Legacy-поведение
  `Access-Control-Allow-Origin: *` — opt-in через `--cors-allow-any`.
- **Retry-After при перегрузке.** Насыщение пула возвращает HTTP 503 с
  заголовком `Retry-After: 30`, а WebSocket-payload `error` теперь содержит
  `retry_after_ms: 30000` — клиенты могут делать back-off без угадывания.
- **Лимит WebSocket-фрейма:** 512 КБ.
- **Пул сессий:** максимум 4 параллельных сессии (настраивается через `--pool-size`).
- **Ограничение аудио-буфера:** 5 с (стриминг) / 10 мин (загрузка файла).
- **Внутренние ошибки санитизируются** — пути и данные модели не утекают клиентам.
- **Таймаут неактивного соединения:** 300 с.

Удалённое развёртывание (TLS + reverse proxy): см. [`docs/deployment.md`](docs/deployment.md).

## Диагностика

| Симптом | Причина | Решение |
|---|---|---|
| `protoc` not found during build | Отсутствует Protocol Buffers compiler | `brew install protobuf` (macOS) или `apt install protobuf-compiler` (Debian/Ubuntu) |
| Загрузка модели зависает или падает | Сеть / доступность HuggingFace | Повторить `gigastt download`; проверить права `~/.gigastt/models/` |
| `Cannot quantize: FP32 encoder not found` | Частичная загрузка | Удалить `~/.gigastt/models/` и повторить `gigastt download` |
| OOM при старте | Pool size слишком большой для доступной RAM | Уменьшить `--pool-size` (по умолчанию 4); каждая сессия загружает полный encoder |
| CoreML не используется на macOS | Собрано без `--features coreml` | Пересобрать: `cargo build --release --features coreml` |
| CUDA недоступен на Linux | Собрано без `--features cuda` или отсутствует CUDA 12+ | Пересобрать: `cargo build --release --features cuda`; проверить `nvidia-smi` |
| WebSocket закрывается с 1008 | Сессия превысила `--max-session-secs` | Увеличить `--max-session-secs` или отправлять более короткие потоки |
| 429 Too Many Requests | Rate limiter включён и bucket исчерпан | Дождаться `Retry-After` или отключить `--rate-limit-per-minute 0` |
| Пустая транскрипция для шумного аудио | Слишком тихий вход или неверный формат | Убедиться в 16-bit PCM; нормализовать уровень; проверить поддерживаемые форматы |

## Тестирование

163 юнит-теста (включая property-based через proptest) + 35 e2e/load/soak-тестов + WER-бенчмарк:

```sh
cargo test --workspace               # 163 юнит-теста (модель не нужна)
cargo clippy --workspace             # Линтер (ноль предупреждений)

# E2E-тесты (требуется модель, последовательно во избежание OOM)
cargo run -p gigastt -- download
cargo test -p gigastt --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit -- --ignored --test-threads=1

# Нагрузочные и стресс-тесты (только локально)
cargo test -p gigastt --test load_test -- --ignored
cargo test -p gigastt --test soak_test -- --ignored
```

## Участие в разработке

См. [CONTRIBUTING.md](CONTRIBUTING.md) — настройка окружения, правила PR и чеклист релиза.

## Лицензия

MIT — см. [LICENSE](LICENSE)

## Благодарности

- [**GigaAM**](https://github.com/salute-developers/GigaAM) от [SberDevices](https://github.com/salute-developers) — модель распознавания речи
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) от [@istupakov](https://github.com/istupakov) — экспорт ONNX-модели и референсная реализация
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — движок инференса
- [**ort**](https://github.com/pykeio/ort) — Rust-биндинги для ONNX Runtime
