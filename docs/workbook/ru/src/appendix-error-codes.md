# Приложение A — Коды ошибок и close-коды

Таблица **симптом → код → что делать**. Полные контракты полей — в
[docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) и
[docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md);
эта страница — оглавление для поваренной книги.

## Сначала пробы

| Проверка | Ожидание | Глава |
|---|---|---|
| `GET /health` | `200`, `model` — `"loading"` или имя головы | [01](01-getting-started.md) |
| `GET /ready` | `200` до первого audio / job | [04](04-streaming-ws.md), [02](02-cli-batch.md) |
| `version` в `/health` | совпадает с развёрнутым бинарём/образом | [06](06-deployment-ops.md) |

**Liveness** — `/health`, **readiness** — `/ready`; не гадайте по «порт открыт».

## REST / SSE / jobs

| HTTP | Code | Типичная причина | Что делать |
|---|---|---|---|
| 400 | `empty_body` | Пустое тело POST | Отправьте байты аудио |
| 400 | `invalid_format` | Плохой `?format=` | `json` / `txt` / `srt` / `vtt` / `md` — [02](02-cli-batch.md) |
| 400 | `unsupported_codec` | Плохой `?codec=` | `pcmu` / `pcma` / `g722` — [03](03-telephony-voip.md) |
| 400 | `invalid_sample_rate` | Нет/неверный rate с raw codec | `sample_rate=8000` или `16000` — [03](03-telephony-voip.md) |
| 400 | `conflicting_modes` | `channels=split` **и** `diarization=true` | Выберите одно — [03](03-telephony-voip.md) |
| 403 | `loopback_only` | Не-loopback `POST /v1/admin/reload` | Вызов с `127.0.0.1` — [06](06-deployment-ops.md#горячая-перезагрузка-модели-без-рестарта) |
| 404 | (без body) / `jobs_disabled` | Jobs API выключен | `--enable-jobs` — [02](02-cli-batch.md) |
| 404 | `job_not_found` | Неизвестный или истёкший job | Сохраняйте результаты на клиенте — [02](02-cli-batch.md) |
| 409 | `job_not_finished` | Result слишком рано | Ждите `done` / поллите status |
| 409 | `job_not_cancellable` | Cancel на terminal job | Игнорируйте / считайте done |
| 409 | `reload_in_progress` | Параллельные admin reload | Подождите и повторите — [06](06-deployment-ops.md) |
| 409 | `punctuation_not_available` | Force `punctuation=true` без модели | Установите punct / `auto` — [07](07-models-and-backends.md) |
| 413 | `payload_too_large` | Тело > `--body-limit-bytes` | Поднимите лимит или jobs — [02](02-cli-batch.md) |
| 422 | `invalid_audio` | Битый/неподдерживаемый контейнер | Таблица форматов — [03](03-telephony-voip.md) |
| 422 | `transcription_error` | Decode ok, inference failed | Логи; INT8 на месте — [07](07-models-and-backends.md) |
| 429 | `rate_limited` | Bucket по IP пуст | `Retry-After`; лимиты / `--trust-proxy` — [06](06-deployment-ops.md) |
| 429 | `queue_full` | Job store полон | Забирайте results / `--jobs-max` — [02](02-cli-batch.md) |
| 503 | `timeout` | Пул насыщен | Backoff; `--pool-size` — [07](07-models-and-backends.md) |
| 503 | `pool_closed` | Shutdown | Переподключение после деплоя — [06](06-deployment-ops.md) |
| 503 | `initializing` | Модель ещё грузится | Поллите `/ready` — [01](01-getting-started.md) |
| 503 | `reload_failed` / `reload_unsupported` | Hot-reload не удался | Почините файлы модели — [06](06-deployment-ops.md) |
| 504 | `inference_timeout` | Прогон > `--inference-timeout-secs` | Поднимите timeout — [02](02-cli-batch.md) |

## WebSocket error codes

| Code | Сессия | Смысл | Что делать |
|---|---|---|---|
| `timeout` | не открылась | Checkout пула | Ждите `retry_after_ms` — [04](04-streaming-ws.md) |
| `pool_closed` | ends | Drain | Reconnect после апгрейда — [06](06-deployment-ops.md) |
| `idle_timeout` | ends (1001) | Нет фреймов | Шлите PCM (тишина ок) — [04](04-streaming-ws.md) |
| `max_session_duration_exceeded` | ends (1008) | `--max-session-secs` | Reconnect; `final` уже сброшен — [04](04-streaming-ws.md) |
| `policy_violation` | ends (1008) | Спам пустыми фреймами | Не шлите empty binaries |
| `inference_timeout` | ends | Инференс чанка слишком долгий | Короче аудио / timeout |
| `inference_error` | continues | Плохой chunk | Почините клиент; сессия жива |
| `inference_panic` | continues, state reset | Panic изолирован | Новая фраза; старые final валидны |
| `configure_too_late` | continues | `configure` после audio | Configure первым — [04](04-streaming-ws.md) |
| `invalid_sample_rate` | continues | Rate не из `supported_rates` | Rate из `ready` |
| `unsupported_protocol_version` | ends | Несовпадение протокола | `1.0` — [04](04-streaming-ws.md) |

## WebSocket close codes

| Code | Когда | Действие клиента |
|---|---|---|
| 1001 Going Away | SIGTERM drain, idle, ping timeout | Reconnect; на shutdown ждите `final` |
| 1008 Policy Violation | max session / empty spam | Caps / поведение клиента |
| 1009 Message Too Big | Фрейм > `--ws-frame-max-bytes` | Меньше PCM-фреймы |
| 1006 Abnormal Closure | Нет close frame | Проверьте процесс; сервер этот код не шлёт |

## Ссылки

- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md)
- [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md)
- [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md)
- [Стриминг](04-streaming-ws.md) · [Деплой](06-deployment-ops.md) · [CLI/jobs](02-cli-batch.md)
