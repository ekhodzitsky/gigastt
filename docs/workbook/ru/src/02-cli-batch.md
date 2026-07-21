# CLI и пакетная обработка

Превращаем папку с записями в транскрипты: разовые прогоны через
`transcribe-batch`, постоянно наблюдаемая папка-сброс через `watch` и
асинхронная очередь через jobs API. Каждый рецепт можно скопировать и
запустить как есть, и каждый заканчивается проверкой результата.

## Сценарий

У вас есть каталог с аудио — записи колл-центра, выпуски подкастов, архив
голосовых заметок — и на выходе нужны текстовые файлы. Иногда это разовая
конвертация архива; иногда записи продолжают поступать, и конвейер должен
работать без присмотра, повторять неудачные попытки и не спотыкаться о
недокопированные файлы.

## Предварительные требования

- Установленный gigastt и скачанная модель (`gigastt download`) — см.
  [Начало работы](01-getting-started.md).
- Папка с аудиофайлами: WAV, MP3, M4A, OGG, FLAC (вложенные папки
  сканируются рекурсивно).
- Больше ничего: `transcribe`, `transcribe-batch` и `watch` — офлайн-команды,
  без сервера и сети.

Что важно знать до написания скриптов: **каждый запуск CLI загружает модель**
(~1–2 с в тёплом состоянии). `transcribe-batch` амортизирует эту стоимость
на всю папку, поэтому предпочитайте его shell-циклу `for` вокруг одиночных
вызовов `transcribe`.

## Рецепт: разовый прогон папки — `transcribe-batch`

Основной рабочий инструмент. Укажите входной и выходной каталоги:

```sh
gigastt transcribe-batch calls/ transcripts/
```

Команда рекурсивно сканирует `calls/`, транскрибирует каждый поддерживаемый
аудиофайл в `--pool-size` воркеров (по умолчанию 2) и пишет
`transcripts/<имя>.txt` и `transcripts/<имя>.json` на каждый входной файл
(по умолчанию `--format txt,json`).

Прогон «как в продакшене» — больше форматов, больше воркеров и политика
исходников:

```sh
gigastt transcribe-batch calls/ transcripts/ \
  --format txt,json,srt \
  --pool-size 4 \
  --retries 2 \
  --move-to calls/done/
```

- `--format` — список через запятую из `txt,json,md,srt,vtt`; по одному
  выходному файлу на формат на входной файл.
- `--pool-size` — параллельные воркеры; каждый стоит ~0,4 ГБ RAM (INT8-
  энкодер), поэтому масштабируйтесь по памяти, а не только по ядрам (см.
  рецепт про производительность ниже).
- `--retries` — дополнительные попытки на файл с коротким бэкоффом
  (200 мс, 400 мс, …). По умолчанию 0 для batch и 2 для watch.
- `--move-to` — перемещать каждый *успешно* транскрибированный исходник в
  указанный каталог. Файлы с ошибкой всегда остаются на месте. Каталог
  move-to исключается из сканирования, поэтому размещение его внутри входной
  папки (`calls/done/`) безопасно и является рекомендуемой раскладкой.
- `--delete-source` — альтернатива `--move-to`: удалять исходники после
  успеха. Несовместимо с `--move-to`.

Как читать отчёт о прогоне. Каждый файл оставляет строку в логе, а прогон
завершается сводкой:

```text
INFO gigastt::batch: done /calls/alpha.wav processed=1 failed=0
WARN gigastt::batch: failed /calls/broken.mp3 error=invalid audio: Unsupported audio format: ...
INFO gigastt: batch finished processed=12 failed=1 skipped=0
```

Коды выхода (скриптуйте по ним, а не по тексту лога):

| Код | Значение |
|---|---|
| `0` | все файлы транскрибированы |
| `1` | хотя бы один файл завершился ошибкой после всех попыток |
| `130` | прервано Ctrl-C — файлы в работе завершаются, остальные пропускаются (`skipped=N` в сводке) |

Пустая входная папка — не ошибка: логируется `no audio files found`, код
выхода 0.

### Проверка результата

```sh
gigastt transcribe-batch calls/ transcripts/ --move-to calls/done/
echo "exit code: $?"        # 0 = чистый прогон, 1 = были ошибки
ls transcripts/             # по <имя>.txt + <имя>.json на каждый исходник
ls calls/done/              # успешно обработанные исходники
ls calls/*.wav 2>/dev/null  # всё, что осталось, завершилось ошибкой — см. строки WARN
```

## Рецепт: живая папка — `watch`

`watch` опрашивает каталог и транскрибирует файлы по мере их появления:

```sh
gigastt watch inbox/ transcripts/ --format txt,json --move-to inbox/done/
```

Чем отличается от `transcribe-batch`:

- **Бэклог пропускается.** Файлы, уже лежащие в папке на момент запуска,
  регистрируются, но *не* транскрибируются (`watching /inbox backlog=3
  poll_ms=1000`). Существующую гору сначала разгребите `transcribe-batch`
  (см. рецепт с обёрткой), а `watch` оставьте для новых поступлений.
- **Settle-защита.** Файл ставится в работу только после того, как его
  размер + mtime не менялись на протяжении `--settle-polls` подряд опросов
  (по умолчанию 2) с интервалом `--poll-interval-ms` (по умолчанию 1000 мс).
  Запись, которую ещё копируют или пишут, никогда не будет подхвачена
  наполовину. Медленные сетевые шары → увеличьте оба параметра.
- **Изменения подхватываются.** Перезапись файла сбрасывает settle-счётчик,
  и новая версия транскрибируется (изменение посреди транскрибации ставит
  файл в очередь повторно после завершения текущего прогона).
- **Ошибки «липкие».** Файл, исчерпавший попытки (по умолчанию 2 для watch),
  помечается failed и не трогается, пока не изменится его содержимое.
- **Мягкая остановка.** Ctrl-C прекращает постановку новых файлов, ждёт
  завершения текущих, печатает `watch stopped processed=N failed=M` и
  выходит с кодом 0 (1, если были ошибки).

### Проверка результата

```sh
# терминал 1
gigastt watch inbox/ transcripts/ --move-to inbox/done/

# терминал 2
cp ~/recordings/sample.wav inbox/
# подождите settle-polls x poll-interval плюс время транскрибации, затем:
ls transcripts/sample.txt        # появился
ls inbox/done/sample.wav         # исходник заархивирован
# в терминале 1: Ctrl-C → "watch stopped processed=1 failed=0"
```

## Рецепт: конвейер-обёртка для inbox (shell)

Стандартный сервис «папка-сброс»: аудио падает в `inbox/`, наружу выходят
транскрипты, успехи архивируются в `done/`, ошибки собираются в `failed/` и
автоматически повторяются при следующем прогоне. Сохраните как
`transcribe-inbox.sh`:

```bash
#!/usr/bin/env bash
# Usage: transcribe-inbox.sh [INBOX] [OUT]
set -uo pipefail

INBOX="${1:-inbox}"
OUT="${2:-transcripts}"
DONE="$INBOX/done"
FAILED="$INBOX/failed"
mkdir -p "$OUT" "$DONE" "$FAILED"

# Requeue previous failures for another attempt.
find "$FAILED" -maxdepth 1 -type f \
  \( -name '*.wav' -o -name '*.mp3' -o -name '*.m4a' -o -name '*.ogg' -o -name '*.flac' \) \
  -exec mv -n {} "$INBOX/" \;

gigastt transcribe-batch "$INBOX" "$OUT" --format txt,json --move-to "$DONE"
rc=$?

# Successes were moved to done/; whatever audio remains at the inbox top
# level failed all retries — collect it for inspection and future requeue.
if [ "$rc" -eq 1 ]; then
  find "$INBOX" -maxdepth 1 -type f \
    \( -name '*.wav' -o -name '*.mp3' -o -name '*.m4a' -o -name '*.ogg' -o -name '*.flac' \) \
    -exec mv -n {} "$FAILED/" \;
  echo "some files failed — collected in $FAILED" >&2
fi
exit "$rc"
```

Запуск по расписанию. Для большинства инбоксов достаточно cron:

```cron
*/15 * * * * /usr/local/bin/transcribe-inbox.sh /srv/stt/inbox /srv/stt/transcripts >> /var/log/stt-batch.log 2>&1
```

Вариант с systemd timer + service-юнитом вместо cron — см.
[Развёртывание и эксплуатация](06-deployment-ops.md).

**Watch + догоняющий batch.** Обе команды складываются в постоянно работающий
конвейер: `watch` обрабатывает поступающие файлы с малой задержкой, а
периодический `transcribe-batch` разгребает стартовый бэклог и всё, что
наблюдатель пометил failed. Обе команды учитывают одно и то же исключение
`--move-to`, поэтому бэклог не обрабатывается дважды. Одна оговорка:
догоняющий прогон, запущенный *пока* наблюдатель работает, может подхватить
файл, который наблюдатель только что поставил в работу, но ещё не переместил —
планируйте прогоны на тихие часы или примите, что файл изредка будет
транскрибирован дважды (его выходные файлы просто перезапишутся).

```sh
# один раз и далее периодически (тихие часы): разобрать бэклог + повторить ошибки
./transcribe-inbox.sh /srv/stt/inbox /srv/stt/transcripts

# постоянно: новые поступления
gigastt watch /srv/stt/inbox /srv/stt/transcripts \
  --format txt,json --move-to /srv/stt/inbox/done/
```

### Проверка результата

```sh
chmod +x transcribe-inbox.sh
cp ~/recordings/*.wav inbox/ && printf 'junk' > inbox/broken.mp3
./transcribe-inbox.sh inbox transcripts; echo "exit: $?"   # 1 — broken.mp3 failed
ls transcripts/   # транскрипты для хороших файлов
ls inbox/done/    # хорошие исходники заархивированы
ls inbox/failed/  # broken.mp3 собран здесь
./transcribe-inbox.sh inbox transcripts   # повторяет ошибочный, снова выходит с 1
```

## Рецепт: конвейер с очередью — jobs API

`watch` покрывает одну машину с общей папкой. Переходите на **jobs API**,
когда производители на других машинах, когда файлы настолько длинные, что
держать синхронный HTTP-запрос неудобно, или когда нужны прогресс и
отмена. Это тот же движок за in-memory FIFO-очередью внутри `gigastt serve`.

Jobs по умолчанию выключены. Включите их (и зарезервируйте инференс-слоты,
чтобы очередь не душила WebSocket/REST-стриминг):

```sh
gigastt serve --enable-jobs --batch-pool-size 1
```

Отправка → опрос → получение:

```sh
# submit (принимает те же query-параметры, что и /v1/transcribe, напр. ?format=srt)
curl -s -X POST http://127.0.0.1:9876/v1/jobs \
  --data-binary @episode.wav
# {"job_id":"019f858a-...","status":"queued","created_at":1784651881.9}

# poll status
curl -s http://127.0.0.1:9876/v1/jobs/019f858a-...
# {"job_id":"...","status":"processing","processed_seconds":12.5,"percent":42}

# fetch the result once status is "done"
curl -s http://127.0.0.1:9876/v1/jobs/019f858a-.../result
# {"text":"...","words":[...],"duration":3512.4}
```

`status` проходит путь `queued` → `processing` → `done` | `failed` |
`cancelled`. Запрос `/result` до `done` возвращает `409 job_not_finished`.
Другие эндпоинты: `DELETE /v1/jobs/{id}` отменяет queued/processing-задачу
(`204`), а `GET /v1/jobs/{id}/events` стримит SSE-прогресс
(`data: {"type":"progress","percent":42,...}`, затем `done`/`failed`).

Минимальный скрипт-драйвер:

```bash
#!/usr/bin/env bash
# submit-and-wait.sh AUDIO_FILE — submit a job and print its transcript.
set -euo pipefail
BASE="${GIGASTT_BASE:-http://127.0.0.1:9876}"

job=$(curl -sf -X POST "$BASE/v1/jobs" --data-binary "@$1" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "job: $job" >&2

while true; do
  status=$(curl -sf "$BASE/v1/jobs/$job" \
           | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  case "$status" in
    done)               break ;;
    failed|cancelled)   echo "job $status" >&2; exit 1 ;;
  esac
  sleep 2
done

curl -sf "$BASE/v1/jobs/$job/result" | python3 -c 'import json,sys; print(json.load(sys.stdin)["text"])'
```

Поведение очереди, которое нужно учитывать:

- `--jobs-retry` (по умолчанию 3) — повторяет только *временные* сбои:
  таймауты инференса и паники воркеров. Файл, который не декодируется,
  падает сразу, без ретраев.
- `--jobs-max` (по умолчанию 100) — когда хранилище полно, submit возвращает
  `429 queue_full` с `Retry-After`. Отступите и отправьте повторно.
- `--jobs-ttl-secs` (по умолчанию 3600) — завершённые/упавшие/отменённые
  задачи вытесняются после TTL. **Забирайте и сохраняйте результаты сразу** —
  хранилище в памяти, поэтому перезапуск сервера теряет и очередь, и
  незабранные результаты.

### Проверка результата

```sh
curl -s http://127.0.0.1:9876/ready     # {"status":"ready",...} перед отправкой
./submit-and-wait.sh episode.wav        # печатает текст транскрипта
# проверка выключенного API: без --enable-jobs любой вызов /v1/jobs возвращает 404
```

## Рецепт: выбор формата выхода

Пять форматов, по файлу на формат на входной файл. Выбирайте по потребителю:

| Формат | Для чего | Примечания |
|---|---|---|
| `txt` | люди, grep, текстовые пайплайны | только текст транскрипта |
| `json` | машины | `{"text", "words": [{"word","start","end","confidence"}], "duration"}` — пословные тайминги и confidence |
| `srt` | видеоредакторы, загрузка на YouTube | кью SubRip, сгруппированные из пословных таймингов |
| `vtt` | веб-плееры | WebVTT-вариант тех же кью |
| `md` | заметки, архивы | YAML-шапка (`duration`, `language`, `speakers`) + транскрипт |

```sh
gigastt transcribe-batch episodes/ out/ --format txt,json        # default pair
gigastt transcribe recording.wav -f srt -o recording.srt         # single file
```

Настройка субтитров (SRT/VTT): `--max-chars-per-line` (по умолчанию 80) и
`--max-words-per-line` (по умолчанию 14) управляют группировкой кью; `0`
отключает ограничение. Для эфирных титров обычно нужны строки короче:

```sh
gigastt transcribe recording.wav -f vtt --max-chars-per-line 42 -o recording.vtt
```

Дополнения Markdown: `--word-timestamps` добавляет пословную таблицу с
таймингами и confidence — удобно для ручной вычитки, шумно для архивов.

Тонкость для скриптов с одиночным `transcribe`: на уровне `info` логи идут в
**stdout** вперемешку с транскриптом. Используйте `-o` для записи
транскрипта в файл или глушите логи глобальным флагом, который ставится
перед подкомандой:

```sh
gigastt --log-level error transcribe recording.wav          # stdout = transcript only
```

Извлечение текста из папки JSON-результатов:

```sh
jq -r '.text' transcripts/*.json
```

### Проверка результата

```sh
gigastt transcribe recording.wav -f srt -o /tmp/check.srt
head -4 /tmp/check.srt
# 1
# 00:00:00,480 --> 00:00:02,160
# Привет, как дела?
jq -r '.duration' transcripts/episode.json    # JSON parses and has fields
```

## Рецепт: необычные входы — телефонные WAV, Opus, raw-потоки

**G.711 / G.722 внутри WAV — работает само.** A-law/μ-law (телефонные
экспорты 8 кГц) и G.722 ADPCM (Asterisk/Cisco/Teams, теги формата
`0x0064`/`0x028F`) декодируются автоматически; batch-обходчик подхватывает
их как любой другой `.wav`.

**OGG/Opus и `.opus` (голосовые Telegram, браузерный MediaRecorder).**
Контейнер определяется по содержимому, поэтому одиночная транскрибация
работает как есть:

```sh
gigastt transcribe voice.opus
```

Но обходчики batch/watch сканируют по расширению (`wav,mp3,m4a,ogg,flac`) и
**не подхватывают `.opus`-файлы**. Переименуйте их в `.ogg` перед прогоном —
содержимое уже является OGG-контейнером, поэтому достаточно простого
переименования:

```sh
for f in inbox/*.opus; do mv "$f" "${f%.opus}.ogg"; done
gigastt transcribe-batch inbox/ transcripts/
```

**Raw-потоки без заголовка** (дампы RTP, Asterisk Monitor raw) не несут
контейнера для определения — объявите кодек и частоту явно:

```sh
gigastt transcribe call.ulaw --codec pcmu --sample-rate 8000
gigastt transcribe call.alaw --codec pcma --sample-rate 8000
gigastt transcribe call.g722 --codec g722 --sample-rate 8000   # 16000 also accepted
```

`--codec` принимает `pcmu` (алиас `ulaw`), `pcma` (алиас `alaw`), `g722` и
требует `--sample-rate`. Всё остальное — WebM, AMR, видео MP4, битый файл —
падает с `invalid audio: Unsupported audio format: ...` (REST:
`422 invalid_audio`).

### Проверка результата

```sh
file recording.wav                 # confirms the container type
gigastt --log-level error transcribe call.ulaw --codec pcmu --sample-rate 8000
echo "exit: $?"                    # 0 = decoded and transcribed
gigastt transcribe call.ulaw --codec pcmu 2>&1 | head -2
# error: the following required arguments were not provided: --sample-rate
```

## Рецепт: производительность и память

Голова `rnnt` на INT8 работает с **RTF ≈ 0,10** на M1 CPU — один воркер
переваривает час аудио примерно за 6 минут. Архив на 100 часов при
`--pool-size 4` закончится примерно за `100 ч × 0,10 / 4 ≈ 2,5 ч`
астрономического времени. Полные измерения, другое железо и цифры WER:
[docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md).

Прежде чем поднимать `--pool-size`, прикиньте память:

- Каждый воркер загружает свою копию энкодера: **~0,4 ГБ resident** с
  INT8-энкодером по умолчанию, **~1,7 ГБ** с FP32. Пул по умолчанию из 2 ≈
  790 МБ RSS.
- Движок не даёт пулу съесть больше половины всей RAM: завышенный
  `--pool-size` **урезается с предупреждением** при загрузке, так что
  проверяйте лог, а не предполагайте, что получили заказанный параллелизм.
- Оставайтесь на INT8 (по умолчанию после авто-квантизации при первом
  запуске): энкодер ужимается с 844 МБ до 215 МБ на диске с деградацией WER
  ~0%, а FP32 учетверяет память на воркера без выигрыша в скорости пакетной
  обработки.
- На CPU-сборках `--encoder-intra-threads` по умолчанию равен числу
  логических CPU, поделённому на размер пула, — правильное значение для
  выделенной batch-машины; крутите только для совместно используемых.

### Проверка результата

```sh
# предупреждение об урезании, если есть, появляется при загрузке:
gigastt transcribe-batch calls/ transcripts/ --pool-size 8 2>&1 | grep -i "pool" | head -3
# per-file throughput in the log: "transcribe complete audio_s=... wall_s=... rtf=0.129"
time gigastt transcribe-batch calls/ transcripts/ --pool-size 4 --move-to calls/done/
```

## Частые ошибки

- **Недокопированные файлы.** `transcribe-batch` транскрибирует то, что
  лежит в папке *сейчас*, включая файл, который ещё копируется, — получите
  ошибку декодирования или обрезанный транскрипт. Производители должны писать
  во временное имя и делать `mv` в inbox (переименование атомарно в пределах
  одной файловой системы). `watch` защищается settle-опросами; batch
  рассчитывает на тихую папку.
- **Случайная повторная обработка.** Без `--move-to`/`--delete-source`
  каждый повторный прогон переделывает всю папку. Для регулярных прогонов
  всегда задавайте политику исходников. Кроме того: `--move-to` схлопывает
  вложенные папки — `a/week1/call.wav` и `a/week2/call.wav` столкнутся в
  одном `done/call.wav` (а про транскрипты прогон предупредит
  `duplicate output ... inputs with equal file stems overwrite each other`).
  Держите имена исходников уникальными.
- **Ожидание параллелизма, которого не получили.** `--pool-size 16` на
  машине с 8 ГБ RAM молча урезается при загрузке (предупреждение в логе).
  Проверяйте стартовый лог и помните, что FP32 учетверяет память на воркера.
- **`invalid audio` / 422 на неподдерживаемом контейнере.** WebM, AMR, видео
  MP4 или битая выгрузка не декодируются. Сначала сконвертируйте
  (`ffmpeg -i in.webm -ar 16000 -ac 1 out.wav`) или, для raw-телефонии,
  объявите `--codec` + `--sample-rate`. Файл `.opus` декодируем, но невидим
  для batch/watch — переименуйте в `.ogg`.
- **Watch «забывает» ошибки.** Файл, исчерпавший попытки, не повторяется,
  пока не изменится его содержимое, а перезапуск наблюдателя регистрирует
  его как бэклог (никогда не обрабатывается). Исправьте или замените файл
  либо направьте на него `transcribe-batch` — для этого и нужен догоняющий
  прогон из рецепта с обёрткой.
- **Результаты задач испаряются.** Хранилище задач в памяти с TTL 1 ч на
  терминальные задачи: перезапуск теряет очередь, а незабранные результаты
  вытесняются. Сохраняйте результаты на клиенте; воспринимайте
  `429 queue_full` как backpressure, а не как ошибку, достойную алерта.

## Ссылки

- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) —
  полный справочник флагов `transcribe`, `transcribe-batch`, `watch`
- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) —
  эндпоинты jobs API, query-параметры, коды ошибок
- [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md) —
  измерения RTF, памяти и WER
- [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) —
  каталог ошибок декодирования и форматов
- [Начало работы](01-getting-started.md) — установка и скачивание модели
- [Развёртывание и эксплуатация](06-deployment-ops.md) — systemd-юниты и
  таймеры для постоянно работающих конвейеров
