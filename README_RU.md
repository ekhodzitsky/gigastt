<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Встраиваемое локальное распознавание русской речи — один бинарник на Rust, без облака, MIT-чистые веса.</strong></p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://docs.rs/gigastt-core"><img src="https://docs.rs/gigastt-core/badge.svg" alt="docs.rs"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  </p>
  <p align="center"><a href="README.md">English</a> | <b>Русский</b></p>
</p>

---

gigastt превращает любую машину в приватный сервер распознавания русской речи — или встраивает тот же движок в Rust-приложение или Android-бинарник. Открытая модель **GigaAM v3** работает полностью локально через ONNX Runtime: без облака, без API-ключей.

```sh
cargo install gigastt && gigastt serve
# WebSocket  ws://127.0.0.1:9876/v1/ws
# REST       http://127.0.0.1:9876/v1/transcribe
```

```sh
$ gigastt transcribe recording.wav
Привет, как дела?
```

## Возможности

- **Стриминг в реальном времени** — инкрементальные partial-результаты по WebSocket; REST + SSE для файлов
- **Встраиваемость** — один статический бинарник, C-ABI FFI `cdylib` для Android/mobile, или крейт `gigastt-core`
- **Точный и маленький** — **2.6% WER** на Golos crowd (голова rnnt), INT8-модель ~225 МБ, real-time на CPU (RTF ~0.11); ускорение CoreML / CUDA / NNAPI
- **Защищённый сервер** — loopback по умолчанию, origin-allowlist, rate-limiting по IP, graceful drain, метрики Prometheus
- **MIT-чистый** — gigastt (MIT) на весах GigaAM v3 (MIT) — пригоден для коммерческих on-device продуктов

## Где это уместно

gigastt — **только русский** и заточен под **встраивание**. Голова rnnt (дефолт с v2.3) даёт **2.6% WER на Golos crowd** — на уровне сильнейших русских движков на чистой начитке — и эта ошибка **чисто акустическая**, не раздутая нормализацией. Для multilingual — whisper.cpp / sherpa-onnx / NVIDIA Parakeet. Ниша gigastt — **самая маленькая русская модель без компромисса по языковой модели**, в обёртке **встраиваемого single-binary / FFI / streaming** сервера с **MIT-чистыми весами**, конкурентная на спонтанной и телефонной речи. Полное честное сравнение vs Vosk 0.54, T-one и Whisper → **[Benchmarks](docs/benchmarks.md)**.

## Документация

| Гайд | Содержание |
|---|---|
| **[API](docs/api.md)** | WebSocket-протокол, REST + SSE, коды ошибок, клиенты (Python/Bun/Go/Kotlin) |
| **[Benchmarks](docs/benchmarks.md)** | WER / RTF / footprint против 6 движков на 4 русских доменах, с оговорками |
| **[Architecture](docs/architecture.md)** | Пайплайн, модель, аппаратное ускорение, INT8, структура проекта |
| **[Android / FFI](ANDROID.md)** | Встраивание через C-ABI на Android |
| **[CLI](docs/cli.md)** · **[Deployment](docs/deployment.md)** · **[Security](SECURITY.md)** · **[Troubleshooting](docs/troubleshooting.md)** | Справочник и эксплуатация |

## Установка

```sh
# Homebrew (macOS arm64 / Linux x86_64)
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt && brew install gigastt

# crates.io — нужен protoc в PATH (brew install protobuf / apt install protobuf-compiler)
cargo install gigastt

# Docker (CUDA: Dockerfile.cuda; вшить модель в образ: --build-arg GIGASTT_BAKE_MODEL=1)
docker build -t gigastt . && docker run -p 9876:9876 gigastt
```

Модель GigaAM v3 (~850 МБ) скачивается автоматически при первом запуске и квантуется в INT8 до ~225 МБ.

> Сборка также тянет prebuilt onnxruntime по сети (ort `download-binaries`); гарантия on-device / без облака покрывает **runtime-инференс**, а не сборку. Air-gapped-сборка — в [Architecture](docs/architecture.md).

## Требования

Rust **1.88+**, `protoc` в `PATH`. macOS 14+ (Apple Silicon, CoreML) или Linux x86_64 (опц. NVIDIA CUDA 12+). ~1.5 ГБ диска, ~790 МБ RAM при дефолтном `--pool-size 2` (~400 МБ на одну сессию). Крейт `gigastt-core` без серверных зависимостей: `gigastt-core = "2.0"`.

## Лицензия

MIT — см. [LICENSE](LICENSE).

> **Данные бенчмарка** под `benchmark/` — **не** MIT: транскрипты OpenSTT (`openstt_*`, CC BY-NC 4.0) и Golos (`golos_*`, Sber Public License) сохраняют свои non-commercial лицензии. См. [`NOTICE`](NOTICE) и [`benchmark/DATA_LICENSE`](benchmark/DATA_LICENSE).

## Благодарности

- [**GigaAM**](https://github.com/salute-developers/GigaAM) от [SberDevices](https://github.com/salute-developers) — модель распознавания
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) от [@istupakov](https://github.com/istupakov) — ONNX-экспорт и референс
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) · [**ort**](https://github.com/pykeio/ort) — движок инференса и Rust-биндинги
