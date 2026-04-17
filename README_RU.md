<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Распознавание русской речи на устройстве с WER 10.4%</strong></p>
  <p align="center">Локальный STT-сервер на базе GigaAM v3 — без облака, без API-ключей, полная приватность</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange" alt="Changelog"></a>
  </p>
  <p align="center"><a href="README.md">English</a> | <b>Русский</b></p>
</p>

---

**gigastt** превращает любой компьютер в сервер распознавания русской речи в реальном времени. Один бинарник, одна команда, точность на уровне лучших решений — всё работает локально.

```sh
cargo install gigastt && gigastt serve
# WebSocket: ws://127.0.0.1:9876/ws
# REST API:  http://127.0.0.1:9876/v1/transcribe
```

## Почему gigastt?

| | gigastt | Whisper large-v3 | Облачные API |
|---|:---:|:---:|:---:|
| **WER (русский)** | **10.4%** | ~18% | 5-10% |
| **Задержка (16с аудио, M1)** | **~700мс** | ~4с | зависит от сети |
| **Стриминг** | WebSocket в реальном времени | только пакетный | по-разному |
| **Приватность** | 100% локально | 100% локально | данные уходят наружу |
| **Стоимость** | бесплатно навсегда | бесплатно | от $0.006/мин |
| **Установка** | `cargo install` | Python + зависимости | API-ключ + биллинг |
| **Размер бинарника** | один бинарник | Python-рантайм | N/A |
| **INT8 квантизация** | авто, 0% потери WER | вручную | N/A |
| **Параллельные сессии** | 4 (настраиваемо) | 1 | без ограничений |

> GigaAM v3 обучена на **700K+ часах** русской речи. Обеспечивает лучшую точность, чем Whisper-large-v3 на русскоязычных бенчмарках, при этом работает быстрее на Apple Silicon и NVIDIA GPU. WER измерен на 993 записях Golos (4991 слово).

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
# Из crates.io
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

Подключение к `ws://127.0.0.1:9876/ws`, отправка PCM16 аудио-фреймов, получение транскрипции в реальном времени.

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
| `/v1/models` | GET | Информация о модели (тип encoder, размер пула, возможности) |
| `/v1/transcribe` | POST | Транскрипция файла, полный JSON-ответ |
| `/v1/transcribe/stream` | POST | Транскрипция файла с SSE-стримингом |
| `/ws` | GET | WebSocket-апгрейд для стриминга в реальном времени |

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

Опциональный квантизированный encoder: в 4 раза меньше, ~43% быстрее, 0% деградации WER (проверено на 993 записях Golos / 4991 слово). Автоматически определяется при запуске.

При сборке с `--features quantize` INT8-encoder создаётся автоматически при первом `download` или `serve` — ручных шагов не требуется.

```sh
# Автоматически (рекомендуется)
cargo install gigastt --features quantize
gigastt serve   # скачивает модель + автоквантизация при первом запуске

# Вручную
gigastt quantize                     # нативная квантизация на Rust
```

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

## Справка по CLI

```
gigastt [ОПЦИИ] <КОМАНДА>

Опции:
  --log-level <LEVEL>    Уровень логирования [по умолчанию: info]

Команды:
  serve        Запустить STT-сервер
  download     Скачать модель (~850 МБ)
  transcribe   Транскрибировать аудиофайл (офлайн)
  quantize     Квантизировать encoder в INT8 (требуется --features quantize)

gigastt serve [ОПЦИИ]
  --port <PORT>          Порт [по умолчанию: 9876]
  --host <HOST>          Адрес привязки [по умолчанию: 127.0.0.1]
  --model-dir <DIR>      Директория модели [по умолчанию: ~/.gigastt/models]
  --pool-size <N>        Параллельные сессии инференса [по умолчанию: 4]

gigastt download [ОПЦИИ]
  --model-dir <DIR>      Директория модели [по умолчанию: ~/.gigastt/models]
  --diarization          Также скачать модель диаризации (требуется --features diarization)

gigastt transcribe [ОПЦИИ] <ФАЙЛ>
  --model-dir <DIR>      Директория модели [по умолчанию: ~/.gigastt/models]
  Поддержка: WAV, M4A, MP3, OGG, FLAC (моно или автомикширование)

gigastt quantize [ОПЦИИ]          # требуется --features quantize
  --model-dir <DIR>      Директория модели [по умолчанию: ~/.gigastt/models]
  --force                Переквантизировать даже если INT8-модель существует
```

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

- Привязка только к `127.0.0.1` по умолчанию (localhost)
- Лимит WebSocket-фрейма: 512 КБ
- Пул сессий: максимум 4 параллельных сессии
- Ограничение аудио-буфера: 5 с (стриминг) / 10 мин (загрузка файла)
- Внутренние ошибки санитизируются — пути и данные модели не утекают клиентам
- Таймаут неактивного соединения: 300 с

## Тестирование

87 юнит-тестов + 24 e2e-теста + нагрузочные и стресс-тесты:

```sh
cargo test                           # 87 юнит-тестов (модель не нужна)
cargo clippy                         # Линтер (ноль предупреждений)

# E2E-тесты (требуется модель, последовательно во избежание OOM)
cargo run -- download
cargo test --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown -- --ignored --test-threads=1

# Нагрузочные и стресс-тесты (только локально)
cargo test --test load_test -- --ignored
cargo test --test soak_test -- --ignored
```

## Лицензия

MIT — см. [LICENSE](LICENSE)

## Благодарности

- [**GigaAM**](https://github.com/salute-developers/GigaAM) от [SberDevices](https://github.com/salute-developers) — модель распознавания речи
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) от [@istupakov](https://github.com/istupakov) — экспорт ONNX-модели и референсная реализация
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — движок инференса
- [**ort**](https://github.com/pykeio/ort) — Rust-биндинги для ONNX Runtime
