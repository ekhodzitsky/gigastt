# GigaSTT — книга рецептов

Сценарные рецепты для [gigastt](https://github.com/ekhodzitsky/gigastt) —
локального сервера русской речи-в-текст на базе GigaAM v3. Каждая глава
устроена одинаково: **сценарий → предпосылки → рецепт → проверка результата →
частые ошибки → ссылки**.

Это **поваренная книга, а не справочник**. Канонические справочники остаются в
[`docs/`](../../../) — книга ссылается на них, а не дублирует.

**Документировано против gigastt 2.14.x.** В install-скриптах предпочитайте
резолв latest-тега (`gh api …/releases/latest`), а не устаревшие миноры.

## Я хочу…

| Цель | Время | Глава |
|---|---|---|
| Первый транскрипт на этой машине | ~5–15 мин | [Начало работы](01-getting-started.md) |
| Пакет папки / watch / async jobs | ~15–30 мин | [CLI и пакетная обработка](02-cli-batch.md) |
| АТС / Opus / raw telephony; стерео-спикеры | ~20 мин | [Телефония и VoIP](03-telephony-voip.md) |
| Live-субтитры или voice-bot по WebSocket | ~30 мин | [Стриминг по WebSocket](04-streaming-ws.md) |
| macOS / Electron / mobile приложение | ~30–60 мин | [Десктоп и встраивание](05-desktop-embedded.md) |
| Прод с метриками, апгрейдами, hot-reload модели | ~45 мин | [Развёртывание и эксплуатация](06-deployment-ops.md) |
| Выбрать голову / INT8 / GPU / pool / punct / hotwords | ~20 мин | [Модели и бэкенды](07-models-and-backends.md) |
| Метки спикеров на mono-встречах | (в гл. 3) | [Телефония — диаризация](03-telephony-voip.md#моно-запись-встречи--спикеры-через-диаризацию) |
| Расшифровать HTTP/WS код ошибки | ~2 мин | [Приложение A — Коды ошибок](appendix-error-codes.md) |
| Air-gapped / offline | ~30 мин | [Приложение B — Offline-чеклист](appendix-offline-checklist.md) |
| Установка на Windows | ~10 мин | [Начало работы — Windows](01-getting-started.md#рецепт-windows-готовый-бинарь) |

## Главы

1. [Начало работы](01-getting-started.md) — установка (macOS/Linux/Windows/Docker/air-gap),
   первая транскрибация. **Новичок · ~5–15 мин**
2. [CLI и пакетная обработка](02-cli-batch.md) — CLI, batch и watch.
   **Новичок · ~15–30 мин**
3. [Телефония и VoIP](03-telephony-voip.md) — G.711/G.722/Opus, АТС,
   stereo split и диаризация. **Средний · ~20 мин**
4. [Стриминг по WebSocket](04-streaming-ws.md) — partials, VAD-эндпоинтинг,
   session caps. **Средний · ~30 мин**
5. [Десктоп и встраивание](05-desktop-embedded.md) — Swift/SPM, sidecar,
   Electron, UniFFI. **Средний · ~30–60 мин**
6. [Развёртывание и эксплуатация](06-deployment-ops.md) — прод, мониторинг,
   апгрейды, admin reload. **Ops · ~45 мин**
7. [Модели и бэкенды](07-models-and-backends.md) — головы, квантование,
   EP, пунктуация/ITN, hotwords. **Средний · ~20 мин**

### Приложения

- [A — Коды ошибок](appendix-error-codes.md) — jump table REST/WS/close
- [B — Offline-чеклист](appendix-offline-checklist.md) — air-gapped операторский список

[Английская версия](../../en/src/README.md) — каноническая; эта книга
зеркалирует её глава в главу.

## Карта документации

Полный инвентарь документации репозитория: что в каждом файле и где он живёт.

### Справочники (канонические — в книге не дублируются)

| Файл | Содержимое | Судьба |
|---|---|---|
| [docs/api.md](../../../api.md) | Справочник HTTP / WebSocket / SSE API | остаётся |
| [docs/asyncapi.yaml](../../../asyncapi.yaml) | AsyncAPI-схема WS-протокола | остаётся |
| [docs/openapi.yaml](../../../openapi.yaml) | OpenAPI-схема REST API | остаётся |
| [docs/cli.md](../../../cli.md) | Справочник CLI (`serve`, `download`, `transcribe`, …) | остаётся |
| [docs/architecture.md](../../../architecture.md) | Обзор архитектуры | остаётся |
| [docs/benchmarks.md](../../../benchmarks.md) | Измерения WER / RTF | остаётся |
| [docs/privacy.md](../../../privacy.md) | Приватность и потоки данных | остаётся |
| [docs/troubleshooting.md](../../../troubleshooting.md) | Таблица «симптом → причина → решение» | остаётся |
| [docs/observability/](../../../observability/) | Алерты Prometheus и дашборд Grafana | остаётся |

### Гайды (актуальные)

| Файл | Содержимое | Судьба |
|---|---|---|
| [docs/deployment.md](../../../deployment.md) | Reverse proxy, TLS, systemd, Docker | остаётся |
| [docs/quickstarts.md](../../../quickstarts.md) | Квикстарты по встраиванию (FFI-биндинги) | остаётся |
| [docs/runbook.md](../../../runbook.md) | Ранбук оператора для production | остаётся |
| [docs/self-hosted-runner.md](../../../self-hosted-runner.md) | Self-hosted CI-раннеры для бенчмарков | остаётся |
| [docs/embedding-packaging.md](../../../embedding-packaging.md) | Линковка и упаковка onnxruntime | остаётся |
| [docs/verifying-releases.md](../../../verifying-releases.md) | Проверка релизных артефактов | остаётся |
| [docs/ane-backend.md](../../../ane-backend.md) | Заметка о бэкенде ANE (Core ML) — живой код `--features ane` | остаётся |
| [docs/candle-backend.md](../../../candle-backend.md) | Заметка о бэкенде Candle/Metal — живой код `--features candle` | остаётся |
| [sdks/go/README.md](../../../../sdks/go/README.md) | Go SDK для WebSocket-клиента | остаётся |
| [sdks/js/README.md](../../../../sdks/js/README.md) | TypeScript SDK для WebSocket-клиента | остаётся |

### Исторические (в архиве)

Завершённые дизайн-документы и планы, сохранённые для истории в
[`docs/archive/`](../../../archive/):

| Файл | Содержимое | Судьба |
|---|---|---|
| [docs/archive/candle-metal-backend-plan.md](../../../archive/candle-metal-backend-plan.md) | План реализации бэкенда Candle/Metal (завершён) | в архиве |
| [docs/archive/candle-metal-backend-design.md](../../../archive/candle-metal-backend-design.md) | Дизайн бэкенда Candle/Metal (замещён поставленным бэкендом) | в архиве |

## Правила для контрибьюторов

- Книга содержит **рецепты**; `docs/api.md`, `docs/cli.md` и схемы
  AsyncAPI/OpenAPI остаются каноническими справочниками. Ссылайтесь на них —
  не копируйте содержимое.
- Каждая команда и пример в главе должны быть проверены перед мерджем.
- Внутри книги (глава ↔ глава, глава ↔ intro) — только **относительные ссылки
  на `.md`**, они работают и на GitHub, и в собранной книге. Ссылки из книги на
  файлы репозитория (`docs/`, `crates/`, …) — только **абсолютные GitHub-URL**,
  относительные на опубликованном сайте ведут в 404. Никакой mdBook-специфичной
  шаблонизации.
- Новые главы следуют структуре [`_template.md`](_template.md).
- **Английская версия — каноническая.** Русская книга (`docs/workbook/ru/`)
  зеркалирует её с идентичными именами файлов; обе версии правятся в одном PR.
- Когда фича меняет документируемую поверхность (CLI-флаги, коды ошибок,
  аудиоформаты), обновляйте главу, оглавление книги `SUMMARY.md` и
  канонические справочники в том же PR — и держите docs-drift gate зелёным:
  `python3 scripts/check-docs-drift.py` (пока advisory в CI; сверяет
  CLI-флаги, коды ошибок WS, аудиоформаты, оглавления mdBook, паритет EN/RU
  и относительные ссылки с кодом).
