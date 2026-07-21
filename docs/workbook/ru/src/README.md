# GigaSTT — книга рецептов

Сценарные рецепты для [gigastt](https://github.com/ekhodzitsky/gigastt) —
локального сервера русской речи-в-текст на базе GigaAM v3. Каждая глава
устроена одинаково: **сценарий → предпосылки → рецепт → проверка результата →
частые ошибки → ссылки**.

Это **поваренная книга, а не справочник**. Канонические справочники остаются в
[`docs/`](../../../) — книга ссылается на них, а не дублирует.

## Главы

1. [Начало работы](01-getting-started.md) — установка, загрузка модели,
   первая транскрибация.
2. [Транскрибация файлов](02-file-transcription.md) — рецепты для CLI,
   пакетного режима и watch-режима.
3. [Потоковая транскрибация](03-streaming.md) — транскрибация в реальном
   времени по WebSocket.
4. [Модели и бэкенды](04-models-and-backends.md) — варианты моделей,
   квантование, провайдеры исполнения, альтернативные бэкенды.
5. [Интеграция с сервером](05-server-integration.md) — REST/SSE/WS API, SDK,
   асинхронные задачи.
6. [Развёртывание и эксплуатация](06-deployment-ops.md) — production-деплой,
   мониторинг и эксплуатация.

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
- Только **относительные ссылки на `.md`**, чтобы они работали и на GitHub, и в
  собранной книге. Никакой mdBook-специфичной шаблонизации.
- Новые главы следуют структуре [`_template.md`](_template.md).
- **Английская версия — каноническая.** Русская книга (`docs/workbook/ru/`)
  зеркалирует её с идентичными именами файлов; обе версии правятся в одном PR.
