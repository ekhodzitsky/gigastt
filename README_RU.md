<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Встраиваемое локальное распознавание русской речи — один бинарник на Rust, без облака, MIT-чистые веса.</strong></p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://codecov.io/gh/ekhodzitsky/gigastt"><img src="https://codecov.io/gh/ekhodzitsky/gigastt/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://docs.rs/gigastt-core"><img src="https://docs.rs/gigastt-core/badge.svg" alt="docs.rs"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  </p>
  <p align="center"><a href="README.md">English</a> | <b>Русский</b></p>
</p>

---

gigastt превращает любую машину в приватный сервер распознавания русской речи — или встраивает тот же движок в Rust-приложение или Android-бинарник. Открытая модель **GigaAM v3** работает полностью локально через ONNX Runtime: без облака, без API-ключей.

## Обзор

| Локально, приватно | Встраивание + стриминг | Точный русский | Маленький и real-time |
|---|---|---|---|
| Без облака и ключей — инференс 100% локальный. MIT-движок на MIT-весах, пригоден для коммерции. | Один бинарник, C-ABI FFI для мобильных или крейт `gigastt-core` — с инкрементальными partial'ами по WebSocket, без Python. | Самый точный на 3 из 4 русских доменов: far-field 4.08%, телефон 18.50%, YouTube 10.91%; ничья на чистой речи. | ~225 МБ INT8, RTF ~0.10 (~10× быстрее реального времени на CPU), холодный старт 0.94 с. |

**WER** чистая 3.55% / far-field 4.08% / телефон 18.50% / YouTube 10.91%  ·  **RTF** ~0.10  ·  **Модель** ~225 МБ INT8  ·  **Холодный старт** 0.94 с  ·  **RAM** ~400 МБ одна сессия / 790 МБ pool-2  ·  **Стриминг** первый partial ~0.78 с

> Голова GigaAM v3 `rnnt`, INT8, Apple M1 CPU, 1000 сэмплов на домен, отказы = 100% WER, 95% bootstrap CI. Все конкуренты замерены одинаково — тем же [харнессом](docs/benchmarks.md), манифестами и нормализацией.

## Как сравнивается

WER (%) на четырёх русских доменах, меньше — лучше, плюс все оси, по которым выбирают движок. gigastt — голова `rnnt`, INT8.

| Движок | Чистая | Far-field | Телефон | YouTube | RTF | Диск | Пик RAM | Холодный старт | Стриминг | Пункт. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| **gigastt** (GigaAM v3 `rnnt`) | 3.55 | **4.08** | **18.50** | **10.91** | 0.10 | ~225 МБ | 790 / ~400 МБ | **0.94 с** | **Да** — инкр. WS | **Да** |
| Vosk 0.54 (Zipformer2) | **2.97** | 6.29 | 22.74 | 17.24 | ~0.03 | 966 МБ | 560 МБ | 1.16 с | Да (сервер) | Аддон |
| T-one (beam + LM) | 6.61 | 14.62 | 21.73 | 23.23 | 0.065 | 138 МБ + 5.5 ГБ LM | — | — | Да (300 мс) | Нет |
| T-one (greedy, без LM) | 7.85 | 17.22 | 22.37 | 26.54 | 0.065 | 138 МБ | 672 МБ | 1.87 с | Да (300 мс) | Нет |
| whisper.cpp (Large v3) | 15.26 | 17.91 | 32.73 | 22.61 | 0.36–0.77 | 2.9 ГБ | — | — | Нет | Да |
| faster-whisper (Large v3) | 15.53 | 17.34 | 24.93 | 15.45 | &gt;1.0 | 2.9 ГБ | 2619 МБ | 8.2 с | Нет | Да |
| faster-whisper-turbo | 14.45 | 18.30 | 26.58 | 15.45 | &gt;1.0 | 1.6 ГБ | 2154 МБ | 6.8 с | Нет | Да |

Условия: Apple M1, CPU EP, INT8/greedy, 1000 сэмплов на домен (чистая речь 992; turbo — срез 300), 95% bootstrap CI. Чистая речь 3.55 (2.9–4.2) пересекается с Vosk 0.54 2.97 (2.4–3.6) — статистическая ничья; победы на far-field / телефоне / YouTube CI-раздельны. RTF &gt; 1.0 = медленнее реального времени на CPU. RAM gigastt — при дефолтном `--pool-size 2` (одна сессия ~400 МБ). «—» = не замерялось. Полная методология и оговорки: [Benchmarks](docs/benchmarks.md).

**Стриминг:** Whisper-движки работают только офлайн — никаких partial'ов во время речи. gigastt отдаёт настоящие инкрементальные partial'ы по WebSocket (первый ~0.78 с на CPU) из одного самодостаточного бинарника без Python; Vosk-server и T-one (чанки 300 мс) тоже стримят. То есть стриминг — чистая победа над Whisper-семейством; а перед Vosk / T-one преимущество в упаковке — инкрементальные partial'ы плюс C-ABI FFI в одном бинарнике, а не в меньшей задержке.

**Пунктуация и регистр:** gigastt выдаёт читаемый русский из коробки — нативно на голове `e2e_rnnt` или маленьким встроенным проходом RuPunct + ITN на дефолтной `rnnt` (`--punctuation` / `--itn`, авто-докачка). Это на уровне Whisper-движков (у них пунктуация нативная) и лучше русских специалистов — Vosk требует отдельный аддон `recasepunc` на 1.6 ГБ, а T-one не даёт пунктуации вовсе.

## Область применения и честные оговорки

Где выигрывают конкуренты и когда gigastt не нужен:

- **Чистая речь — ничья, не победа** — gigastt 3.55% (2.9–4.2) vs Vosk 0.54 2.97% (2.4–3.6); CI пересекаются, точечная оценка Vosk чуть впереди.
- **Русский в первую очередь, узкий multilingual** — дефолтные головы `rnnt` / `e2e_rnnt` только русские; опциональные `ml_ctc` / `ml_ctc_large` добавляют лишь ru/en/kk/ky/uz. Для реальной широты языков — Vosk (20+) или whisper.cpp / faster-whisper / sherpa-onnx (~99). gigastt — специалист.
- **Не лидер по скорости** — Vosk (RTF ~0.03) и T-one (~0.06) быстрее; gigastt (~0.10) уверенно real-time, но не самый быстрый.
- **По пиковой RAM при дефолтном `--pool-size 2` (790 МБ) проигрывает** Vosk 0.54 (560 МБ) и T-one greedy (672 МБ); конкурентна только одна сессия (~400 МБ) — для экономии ставьте `--pool-size 1`.
- **Стриминг — буферизованный/чанковый** поверх офлайн RNN-T, не нативно-стримящая акустическая модель; ~0.78 с до первого partial — не заявка на минимальную задержку.
- **Пересечение с обучающими данными** — GigaAM v3 обучена в основном на Golos; срезы Golos / OpenSTT, скорее всего, пересекаются с обучающим распределением, так что числа — оптимистичная оценка на in-distribution данных, а не WER на невиданных.

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

## Быстрый старт

```sh
$ gigastt transcribe recording.wav
Привет, как дела?

# Или сервер — WebSocket + REST + SSE на одном порту (только loopback):
$ gigastt serve
# WebSocket  ws://127.0.0.1:9876/v1/ws
# REST       http://127.0.0.1:9876/v1/transcribe
```

## Возможности

| Возможность | Поддержка |
|---|---|
| Головы | `rnnt` (34-токенный char, дефолт — ниже всех WER) · `e2e_rnnt` (1025-токенный BPE, пунктуация / регистр / ITN встроены) · `ml_ctc` / `ml_ctc_large` (GigaAM Multilingual charwise-CTC, 220M / 600M, 71-токенный multilingual char — ru/en/kk/ky/uz) |
| Постобработка | опциональные пунктуация, регистр и русский ITN — нативно на `e2e_rnnt` или встроенный проход RuPunct + ITN на `rnnt` (авто-докачка; `--punctuation` / `--itn`), переопределяемо на каждый запрос (`?punctuation=` / `?itn=` / `?vad=`) |
| Доставка | статический бинарник · C-ABI FFI `cdylib` (Android / mobile) · крейт `gigastt-core` (без серверных зависимостей) |
| Провайдеры исполнения | CPU (любая платформа) · CoreML / Neural Engine (macOS ARM64) · CUDA 12+ (Linux x86_64) · NNAPI (Android) |
| Стриминг | инкрементальные partial'ы по WebSocket · REST + SSE для файлов · один порт 9876 |
| Аудио на вход | WAV · M4A/AAC · MP3 · OGG/Vorbis · FLAC (авто-микс в моно) |
| Асинхронные задачи | Очередь для длинных файлов / batch-распознавания через `/v1/jobs` (включается `--enable-jobs`): submit, poll, отмена, SSE-прогресс, retry и TTL-евикция |
| Экспорт | JSON · TXT · SRT · VTT · Markdown — пословные тайминги + confidence или посегментно (`?segments=true` JSON, `### [mm:ss]` Markdown) |
| Защита сервера | loopback по умолчанию · origin-allowlist · rate-limiting по IP · graceful drain · Prometheus `/metrics` на отдельном порту · loopback-only горячая перезагрузка модели (`POST /v1/admin/reload`) |

## Документация

| Гайд | Содержание |
|---|---|
| **[API](docs/api.md)** | WebSocket-протокол, REST + SSE, коды ошибок, клиенты (Python/Bun/Go/Kotlin) |
| **[Benchmarks](docs/benchmarks.md)** | WER / RTF / footprint против 6 движков на 4 русских доменах, с оговорками |
| **[Architecture](docs/architecture.md)** | Пайплайн, модель, аппаратное ускорение, INT8, структура проекта |
| **[Android / FFI](ANDROID.md)** | Встраивание через C-ABI на Android |
| **[CLI](docs/cli.md)** · **[Deployment](docs/deployment.md)** · **[Security](SECURITY.md)** · **[Troubleshooting](docs/troubleshooting.md)** | Справочник и эксплуатация |

## Требования

Rust **1.88+**, `protoc` в `PATH`. macOS 14+ (Apple Silicon, CoreML) или Linux x86_64 (опц. NVIDIA CUDA 12+). ~1.5 ГБ диска, ~790 МБ RAM при дефолтном `--pool-size 2` (~400 МБ на одну сессию). Крейт `gigastt-core` без серверных зависимостей: `gigastt-core = "2.11"`.

## Лицензия

MIT — см. [LICENSE](LICENSE).

> **Данные бенчмарка** под `benchmark/` — **не** MIT: транскрипты OpenSTT (`openstt_*`, CC BY-NC 4.0) и Golos (`golos_*`, Sber Public License) сохраняют свои non-commercial лицензии. См. [`NOTICE`](NOTICE) и [`benchmark/DATA_LICENSE`](benchmark/DATA_LICENSE).

## Благодарности

- [**GigaAM**](https://github.com/salute-developers/GigaAM) от [SberDevices](https://github.com/salute-developers) — модель распознавания
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) от [@istupakov](https://github.com/istupakov) — ONNX-экспорт и референс
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) · [**ort**](https://github.com/pykeio/ort) — движок инференса и Rust-биндинги
