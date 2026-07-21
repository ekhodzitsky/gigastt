# Развёртывание и эксплуатация

## Сценарий

Вы — администратор, который запускает gigastt на сервере, а не на ноутбуке.
Маршрут этой главы: **установить → ограничить → наблюдать → обновлять** —
один управляемый сервис (systemd или Docker), метрики в Prometheus/Grafana,
алерты на важные режимы отказа и процедура обновления, не обрывающая живые
сессии транскрибации.

Каждый рецепт заканчивается шагом **«Проверить»**. Флаги сверены с
`gigastt serve --help`; полный справочник флагов живёт в
[docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)
и здесь не повторяется.

## Предпосылки

- gigastt установлен (бинарник, пакет или образ) — см.
  [Начало работы](01-getting-started.md).
- Linux-хост с **4+ ГБ RAM**: при `--pool-size 2` по умолчанию с INT8-энкодером
  RSS составляет ~790 МиБ; оставьте запас под ОС и пики запросов.
- Для пути с systemd: systemd 241 или новее (любой современный дистрибутив,
  включая Astra Linux, RED OS, ALT) и root-доступ.
- Для пути с Docker: Docker 20.10+; NVIDIA Container Toolkit — только для
  CUDA-варианта.
- Модель либо скачивается один раз (~850 МБ FP32, автоматически квантуется в
  INT8 при первом запуске), либо предустановлена из офлайн-бандла / deb с
  моделью.

## Рецепт

### Docker

Каждый тегированный релиз публикует мультиархитектурные образы в GHCR —
предпочтительнее тянуть готовое, а не собирать:

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:2.13.0        # CPU, linux/amd64 + linux/arm64
docker pull ghcr.io/ekhodzitsky/gigastt:2.13.0-cuda   # CUDA, linux/amd64
```

Закрепляйте конкретный тег для воспроизводимых развёртываний; `:latest` /
`:cuda` — плавающие.

Запускайте с именованным томом, чтобы модель ~850 МБ (и автоматически
сгенерированный INT8-энкодер) переживали замену контейнера:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:2.13.0
```

Примечания:

- Команда образа по умолчанию — `serve --port 9876 --host 0.0.0.0
  --bind-all` (это нужно контейнерной сети); `-p 127.0.0.1:9876:9876`
  оставляет доступ с хоста только на loopback. TLS-прокси ставится впереди
  точно так же, как при установке без контейнера.
- Контейнер работает под непривилегированным пользователем `gigastt`;
  каталог моделей внутри — `/home/gigastt/.gigastt/models`, именно он
  монтируется томом.
- В образ встроен `HEALTHCHECK` на `/health`, так что `docker ps` покажет
  `healthy`, как только порт начнёт отвечать. Во время первичного скачивания
  модели и INT8-квантования (~2 мин) `/health` отвечает `200` с
  `model:"loading"`, а `/ready` —
  `503 {"status":"not_ready","reason":"initializing"}` — пропускайте трафик
  по `/ready`, а не по `/health`.
- **Baked-образ** (нулевой холодный старт, +~850 МБ): соберите локально с
  моделью внутри — `docker build --build-arg GIGASTT_BAKE_MODEL=1 -t
  gigastt:baked .`
- **CUDA**: `docker run --gpus all -p 127.0.0.1:9876:9876
  ghcr.io/ekhodzitsky/gigastt:2.13.0-cuda` (требуется NVIDIA Container
  Toolkit; при отсутствии GPU бинарник откатывается на CPU).

**Проверить:**

```sh
curl -s http://127.0.0.1:9876/ready
# {"status":"ready","pool_available":2,"pool_total":2}
curl -s http://127.0.0.1:9876/health
# {"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.13.0","punctuation":true,"itn":true}
```

### Установка без сети (замкнутый контур)

Для хостов без доступа в интернет каждый релиз публикует самодостаточный
tarball под каждую Linux-цель — бинарник + предквантованная INT8-модель
`rnnt` + модель пунктуации + systemd-юнит + установщик — и два Debian-пакета
(`gigastt_<ver>_<arch>.deb` + `gigastt-model-int8_<ver>_all.deb`). Полный
состав бандла приведён в
[README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md)
и здесь не повторяется.

На машине с сетью скачайте файлы и **проверьте их до** переноса в контур
(зачем и от каких угроз:
[docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md)):

```sh
gh release download v2.13.0 -R ekhodzitsky/gigastt \
    -p 'gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz' \
    -p 'gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz.sha256' \
    -p 'gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz.minisig'
sha256sum -c gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz.sha256
minisign -Vm gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz -p gigastt.pub
gh attestation verify gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz \
    --repo ekhodzitsky/gigastt
```

На целевом хосте:

```sh
tar xf gigastt-2.13.0-offline-x86_64-unknown-linux-gnu.tar.gz
cd gigastt-2.13.0-offline
sudo ./install.sh    # verifies SHA256SUMS.txt, then installs binary + models + unit
sudo systemctl enable --now gigastt
```

Альтернатива для Debian-семейства:

```sh
sudo dpkg -i gigastt_2.13.0_amd64.deb gigastt-model-int8_2.13.0_all.deb
sudo systemctl enable --now gigastt
```

Модель уже в INT8 — ни скачивания, ни квантования, ни сети. Установленный
юнит выставляет `GIGASTT_OFFLINE=1` через `/etc/gigastt/gigastt.env`, поэтому
любой путь кода, который попытался бы скачать модель (включение `--vad`,
диаризация, альтернативная голова распознавания), **падает быстро с ошибкой,
называющей нужный файл**, вместо зависания на connect timeout. Чтобы добавить
опциональные модели позже, выполните `gigastt download` на машине с сетью и
скопируйте файлы в `/usr/share/gigastt/models/`.

Типовые ошибки:

- `install.sh` прерывается с `sha256sum: WARNING: 1 computed checksum did NOT
  match` — tarball повреждён при переносе в замкнутый контур. Проверьте
  внешний `.sha256`, скопируйте заново и повторите; частичной установки не
  происходит.
- Ошибка офлайн-режима, называющая отсутствующий файл (например, модель VAD
  после включения `--vad`) — этой модели нет в бандле; скачайте её на машине
  с сетью и скопируйте.

**Проверить:**

```sh
systemctl is-active gigastt
# active
curl -s http://127.0.0.1:9876/health
# {"status":"ok",...} — served immediately, the model is pre-installed
```

### systemd-сервис

Усиленный юнит лежит в
[packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd)
и устанавливается как deb-пакетом, так и офлайн-бандлом. Ключевые свойства
(сам юнит короткий и с комментариями — полный список читайте в нём):

- Запуск под непривилегированным пользователем `gigastt`; модели в
  `/usr/share/gigastt/models` доступны ему только на чтение.
- Прослушивание только loopback (`127.0.0.1:9876`); наружу API выставляется
  через reverse proxy.
- `Restart=on-failure`, `RestartSec=5` — падение перезапускается, чистый
  `systemctl stop` — нет.
- Набор харденинга, совместимый с systemd 241 (`ProtectSystem=strict`,
  `NoNewPrivileges`, `PrivateTmp`, …), поэтому юнит работает без изменений на
  Astra Linux, RED OS и ALT.
- Переопределения живут в `/etc/gigastt/gigastt.env` (переменные `GIGASTT_*`,
  `RUST_LOG`) и подхватываются через `EnvironmentFile`.

Логи идут в журнал:

```sh
journalctl -u gigastt -f          # follow
journalctl -u gigastt -n 100      # recent
```

Уровень логирования меняется правкой `/etc/gigastt/gigastt.env`
(`RUST_LOG=gigastt=debug`), затем `sudo systemctl restart gigastt`.

Флаги меняйте через drop-in — никогда не правьте поставляемый юнит (обновление
пакета его перезапишет). `ExecStart` нужно сначала очистить, потом задать
заново:

```ini
# sudo systemctl edit gigastt
[Service]
ExecStart=
ExecStart=/usr/bin/gigastt serve --model-dir /usr/share/gigastt/models --punct-model-dir /usr/share/gigastt/models/punct --metrics
```

`systemctl restart gigastt` шлёт `SIGTERM`; сервер дренирует живые
WebSocket/SSE-сессии — каждый клиент получает кадр `Final` +
`Close(1001 Going Away)` — в течение `--shutdown-drain-secs` (по умолчанию
10 с), что с запасом укладывается в стандартный стоп-таймаут systemd 90 с.
Как использовать это при обновлении версий:
[Обновление и откат](#обновление-и-откат) ниже.

**Проверить:**

```sh
systemctl status gigastt --no-pager
curl -s http://127.0.0.1:9876/health
```

### Наблюдаемость

Метрики включаются опционально и отдаются на **отдельном слушателе** — никогда
на порту API, поэтому они находятся вне CORS-allowlist и per-IP
rate-лимитера:

```sh
gigastt serve --metrics                                  # http://127.0.0.1:9090/metrics
gigastt serve --metrics --metrics-listen 127.0.0.1:9100  # custom port
```

Держите слушатель на loopback, если только ваш Prometheus не на другом хосте —
и даже тогда привязывайте его к доверенному интерфейсу, никогда к публичному.

Минимальная проводка Prometheus (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: gigastt
    static_configs:
      - targets: ["127.0.0.1:9090"]

rule_files:
  - /etc/prometheus/rules/gigastt-alerts.yml   # copy of docs/observability/alerts.yml
```

Метрики, которые важны (все с префиксом `gigastt_`):

| Метрика | Значение |
|---|---|
| `gigastt_http_requests_total` | Запросы по path/method/status — доля 5xx, 503 |
| `gigastt_http_request_duration_seconds` | Гистограмма HTTP-латентности (p50/p95/p99) |
| `gigastt_pool_available` / `gigastt_pool_waiters` | Свободные триплеты инференса против ожидающих вызовов — сигнал насыщения |
| `gigastt_pool_timeouts_total` | Таймауты checkout → клиенты получили 503 + `Retry-After` |
| `gigastt_inference_timeouts_total` | Запуски, прерванные `--inference-timeout-secs` |
| `gigastt_inference_duration_seconds` | Гистограмма латентности инференса |
| `gigastt_ws_active_connections` | Живые WebSocket-сессии |
| `gigastt_rate_limit_rejections_total` | Ответы 429 от per-IP лимитера |
| `gigastt_batch_pool_available` / `gigastt_batch_pool_waiters` | Те же метрики пула для раздела `--batch-pool-size` |

Готовые артефакты — импортируйте, не изобретайте:

- [docs/observability/alerts.yml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/alerts.yml)
  — правила Prometheus: 5xx выше 5%, `gigastt_pool_available == 0` в течение
  1 мин, p95 выше 10 с, устойчивые таймауты пула, падение health-пробы
  (blackbox exporter).
- [docs/observability/dashboard.json](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/dashboard.json)
  — дашборд Grafana (Dashboards → Import): частота запросов, латентность,
  5xx, доступность пула, активные WebSocket, длительность инференса,
  отказы rate-лимитера.

Что алертить на практике: **насыщение пула** (`gigastt_pool_available == 0`
устойчиво — клиенты получают 503), **долю 5xx** и **RAM** на уровне узла
(gigastt не экспортирует метрику собственного RSS; используйте node_exporter
или cAdvisor).

Логи: env-фильтр `tracing` через `RUST_LOG` (по умолчанию `gigastt=info`;
`gigastt=debug` для разбора). Логи содержат метаданные запросов — длительности,
число слов — но никогда текст транскриптов
([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Проверить:**

```sh
curl -s http://127.0.0.1:9876/ready > /dev/null   # samples the pool gauges once
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
# gigastt_pool_available 2
```

### Безопасность по умолчанию

Значения по умолчанию — это уже усиленная конфигурация, поэтому рецепт в
основном о том, как её не ослабить:

- **Привязка к loopback.** `serve` отказывается слушать не-loopback адреса,
  пока не задан `--bind-all` / `GIGASTT_ALLOW_BIND_ANY=1`. Удалённый доступ =
  TLS-терминирующий reverse proxy на том же хосте (конфиги Caddy/nginx:
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md)).
- **Origin-allowlist.** Loopback-источники разрешены всегда; любой другой
  `Origin` нужно перечислить через `--allow-origin` (повторяемый флаг, точное
  совпадение). `--cors-allow-any` — только для разработки. Неразрешённые
  источники получают `403`.
- **Лимиты запросов.** `--body-limit-bytes` (по умолчанию 50 МиБ),
  `--ws-frame-max-bytes` (512 КиБ), `--idle-timeout-secs` (300),
  `--max-session-secs` (3600), `--inference-timeout-secs` (600),
  `--pool-checkout-timeout-secs` (30) — при насыщении пула 503 +
  `Retry-After`.
- **Rate-лимитинг** (опционально): `--rate-limit-per-minute N` с
  `--rate-limit-burst` → `429` + `Retry-After`. За прокси он работает
  по-клиентски, только если прокси *перезаписывает* `X-Forwarded-For` и задан
  `--trust-proxy` — копируйте сниппеты прокси из
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#rate-limiter--x-forwarded-for)
  дословно.
- **Целостность модели.** Скачивания проверяются по SHA-256 и атомарно
  переименовываются (`.partial` → финальный); повреждённый файл никогда не
  попадает в каталог моделей.
- **Верификация релизов.** У каждого артефакта релиза есть `.sha256`-сайдкар +
  `SHA256SUMS.txt`, подпись minisign, CycloneDX SBOM и SLSA-провенанс сборки.
  Проверяйте перед установкой —
  [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md).
  Минимальный ритуал:

```sh
minisign -Vm gigastt-2.13.0-x86_64-unknown-linux-gnu.tar.gz -p gigastt.pub
gh attestation verify gigastt-2.13.0-x86_64-unknown-linux-gnu.tar.gz \
    --repo ekhodzitsky/gigastt
```

- **Приватность.** Нет телеметрии, нет исходящих соединений после разового
  скачивания модели, транскрипты не логируются
  ([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Проверить:**

```sh
ss -ltnp | grep 9876
# tcp LISTEN 0 ... 127.0.0.1:9876 ...   (loopback only)
curl -s -o /dev/null -w '%{http_code}\n' \
    -H 'Origin: https://attacker.example' http://127.0.0.1:9876/v1/models
# 403
```

### Обновление и откат

Закрепляйте то, что разворачиваете (тег образа, версию deb), чтобы обновление
было осознанным и обратимым шагом. Каталог моделей — это состояние: он
переживает обновления, движок сам определяет установленную голову
распознавания, и **никакого молчаливого перекачивания** при смене бинарника
не происходит.

Docker:

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:2.13.1
docker stop --time 15 gigastt && docker rm gigastt
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:2.13.1
```

`docker stop` шлёт `SIGTERM`; `--time 15` даёт окну дренажа
(`--shutdown-drain-secs`, по умолчанию 10 с) завершиться до `SIGKILL` —
стандартные 10 с Docker соревнуются с дренажом. Клиенты получают `Final` +
`Close(1001)` и переподключаются; короткие REST-загрузки в полёте, возможно,
придётся повторить.

systemd / deb:

```sh
sudo dpkg -i gigastt_2.13.1_amd64.deb
sudo systemctl restart gigastt
journalctl -u gigastt -f    # expect a clean drain, no "Drain window expired"
```

В Kubernetes то же правило действует со стороны оркестратора:
`terminationGracePeriodSeconds` ≥ `shutdown_drain_secs + 5` (полный манифест:
[docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#graceful-shutdown--session-caps)).

**Откат.** Разверните предыдущий тег или пакет — набор моделей на диске не
менялся, поэтому старый бинарник стартует на тех же файлах:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:2.13.0
# or: sudo dpkg -i gigastt_2.13.0_amd64.deb && sudo systemctl restart gigastt
```

Если регрессия дренажа ломает ваших WebSocket-клиентов после обновления,
аварийный выход — `--shutdown-drain-secs 0` (прижимается к 1 с); полная
таблица симптомов —
[docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md).

**Проверить** (после каждого обновления или отката):

```sh
curl -s http://127.0.0.1:9876/health
# "version" is the release you deployed; "model"/"variant" are unchanged
curl -s http://127.0.0.1:9876/ready
```

## Проверка результата

Сквозной смоук после любого рецепта выше:

```sh
systemctl is-active gigastt || docker ps --filter name=gigastt --format '{{.Status}}'
curl -s http://127.0.0.1:9876/health     # status ok, expected version
curl -s http://127.0.0.1:9876/ready      # ready, pool_available >= 1
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
```

Затем транскрибируйте один короткий файл через тот API, который реально
выставлен (REST-рецепты:
[CLI и пакетная обработка](02-cli-batch.md); проверка через CLI:
[Начало работы](01-getting-started.md)).

## Частые ошибки

- **OOM — контейнер или сервис убит.** RSS растёт вместе с `--pool-size`:
  INT8-энкодер — ~400 МиБ на триплет, ~790 МиБ при пуле 2 по умолчанию;
  FP32-энкодер примерно в 4 раза больше (никогда не передавайте
  `--skip-quantize` в production). На машине с 4 ГБ держите `--pool-size` в
  пределах 1–2; `--pool-min-size 1` позволяет серверу подняться на
  деградированном пуле вместо падения при нехватке памяти. Если Kubernetes
  сообщает `OOMKilled`, уменьшите пул или поднимите лимит пода — подробности в
  [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md).
- **503 `timeout` под нагрузкой.** Все триплеты заняты, и вызывающий дождался
  конца `--pool-checkout-timeout-secs` (30 с): REST получает `503` +
  `Retry-After`, WebSocket — ошибку с `retry_after_ms`. Это противодавление,
  а не баг — поднимите `--pool-size`, отделите пакетную работу через
  `--batch-pool-size` и следите за `gigastt_pool_waiters` /
  `gigastt_pool_timeouts_total`.
- **`/metrics` недоступен с хоста Prometheus.** Так задумано: слушатель по
  умолчанию — `127.0.0.1:9090`. Направьте скрейпер на сам хост gigastt или
  осознанно перепривяжите порт через `--metrics-listen` на доверенном
  интерфейсе — никогда на публичном. Скрейпер, по-прежнему смотрящий на
  `:9876/metrics`, получит 404: метрики убраны с порта API.
- **Флапающие readiness-пробы при первом запуске.** Первый запуск скачивает
  ~850 МБ и квантует (~2 мин). Всё это время `/health` возвращает `200` с
  `model:"loading"`, но `/ready` возвращает `503 initializing`. Если ваш
  балансировщик маршрутизирует по `/health`, ранние клиенты получат 503 —
  пробуйте `/ready` или предустановите / запеките модель, чтобы окно
  исчезло.
- **Rate-лимитер штрафует всех за прокси.** Без `--trust-proxy` — и без
  прокси, перезаписывающего, а не дописывающего `X-Forwarded-For` — все
  клиенты делят один бакет, ключованный адресом прокси. Симптомы и точная
  конфигурация прокси:
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#rate-limiter--x-forwarded-for).

## Ссылки

- [Начало работы](01-getting-started.md) — установка и первая транскрибация
- [CLI и пакетная обработка](02-cli-batch.md) — рецепты REST / SSE / jobs
- [Стриминг по WebSocket](04-streaming-ws.md) — паттерны WebSocket-протокола
- [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md) — reverse proxy, TLS, манифесты Kubernetes
- [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md) — симптом → причина → аварийный выход
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — полный справочник флагов `serve`
- [docs/observability/alerts.yml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/alerts.yml) — правила алертинга Prometheus
- [docs/observability/dashboard.json](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/dashboard.json) — дашборд Grafana
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) — minisign, SBOM, SLSA-провенанс
- [docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md) — какие данные куда движутся
- [packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd) — юнит + env-файл
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) — состав офлайн-бандла
