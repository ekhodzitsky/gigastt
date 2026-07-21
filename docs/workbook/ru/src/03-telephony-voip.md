# Телефония и VoIP: G.711, G.722, Opus и записи PBX

## Сценарий

Вы обслуживаете колл-центр или интегрируете АТС (Asterisk, FreeSWITCH, Cisco,
Teams), и записи звонков — ваш основной источник аудио. Перед вами папка,
в которой намешаны `wav49`-файлы, WAV с G.711/G.722, сырые `.ulaw`-дампы,
снятые с RTP, и пара голосовых из Telegram — и нужны транскрипты без ручной
конвертации каждого файла.

gigastt декодирует большинство телефонных форматов сам: G.711 A-law/μ-law
в WAV, G.722 в WAV (оба зарегистрированных тега формата), OGG/Opus, а также
сырые потоки без заголовков — через явную подсказку кодека. Эта глава ведёт
от «папки со странными файлами» к рабочим транскриптам, включая разделение
спикеров по каналам для стереозаписей.

## Требования

- Установленный gigastt и скачанная модель — см.
  [Начало работы](01-getting-started.md).
- Запущенный сервер для REST-рецептов (`gigastt serve`, по умолчанию
  `http://127.0.0.1:9876`). CLI-рецепты работают офлайн, сервер не нужен.
- `ffprobe`/`ffmpeg` — только для определения формата и двух запасных
  вариантов с конвертацией (wav49, G.729). Для поддерживаемых форматов
  не нужны.

## Рецепт

### Шаг 0 — определяем, что за файл перед нами

Выгрузки АТС редко говорят, что они такое. Две команды снимают вопрос:

```sh
file recording.wav
ffprobe -v error -show_entries stream=codec_name,codec_tag_string,sample_rate,channels \
  -of default=noprint_wrappers=1 recording.wav
```

Сверьте вывод с таблицей:

| ffprobe / `file` говорит | Что это | Что делать |
|---|---|---|
| `codec_name=pcm_alaw` или `pcm_mulaw`, 8000 Hz | G.711 в WAV | отправлять как есть |
| `codec_name=adpcm_g722`, тег `[0x0064]` или `[0x028f]` | G.722 в WAV | отправлять как есть |
| `codec_name=gsm_ms` | wav49 (GSM 06.10 в WAV) | сначала конвертировать — см. Asterisk ниже |
| `codec_name=opus` в контейнере Ogg | Opus (Telegram, MediaRecorder) | отправлять как есть |
| ffprobe падает с `Invalid data found when processing input`, `file` говорит `data` | сырой поток без заголовков | объявить кодек — см. RTP-дамп ниже |

Оба тега G.722 — один и тот же кодек от разных писалок: `0x0064` приходит
из выгрузок в стиле SBC/Asterisk, `0x028F` — из инструментов на базе ffmpeg.
gigastt принимает оба.

### Записи Asterisk Monitor (wav49, G.722 WAV, сырые потоки)

Что пишут `Monitor()`/`MixMonitor`, зависит от настроенного формата:

- **`wav`** — обычный PCM16 WAV. Отправлять как есть.
- **`wav49`** — GSM 06.10 в WAV-контейнере (`codec_name=gsm_ms`). gigastt
  не декодирует GSM, поэтому один раз конвертируем в PCM16 WAV:

  ```sh
  ffmpeg -y -i call.wav -ar 16000 -ac 1 -c:a pcm_s16le call_16k.wav
  ```

  Проверка: `ffprobe call_16k.wav` показывает `codec_name=pcm_s16le`,
  `sample_rate=16000`. Дальше транскрибируем `call_16k.wav` как обычный WAV.
- **`ulaw` / `alaw` / `g722`** — сырые потоки без заголовков (контейнеру
  нечего «понюхать»). Кодек объявляем явно:

  ```sh
  # REST — алиасы кодеков: pcmu=ulaw, pcma=alaw
  curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=pcmu&sample_rate=8000" \
    -H "Content-Type: application/octet-stream" --data-binary @call.ulaw

  # CLI
  gigastt transcribe call.ulaw --codec pcmu --sample-rate 8000
  ```

  Проверка: HTTP 200 и транскрипт, совпадающий со звонком. Если перепутать
  компандирование (A-law вместо μ-law), запрос всё равно вернёт 200, но текст
  будет мусором — поменяйте имя кодека и повторите.

### Выгрузки Cisco и Teams (G.722 WAV)

Телефония Cisco и инструменты вокруг Teams обычно отдают G.722 в
WAV-контейнере — `codec_name=adpcm_g722`, тег `0x0064` или `0x028F`.
Контейнер сам объявляет кодек, поэтому флаги не нужны:

```sh
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" --data-binary @call_g722.wav

gigastt transcribe call_g722.wav
```

Проверка: HTTP 200 с JSON-текстом, либо транскрипт в stdout для CLI.

Типичный сбой здесь — `422`: сервер не смог декодировать загрузку. Это значит,
что файл не тот, за кого себя выдаёт (внутри G.729 или проприетарная
обёртка вендора), либо файл обрезан. Вернитесь к шагу 0 и посмотрите, что
на самом деле говорит ffprobe.

### Голосовые Telegram и WhatsApp (Opus)

Голосовые Telegram (Bot API отдаёт `voice.oga`), голосовые WhatsApp (`.ogg`)
и записи браузерного MediaRecorder (`.opus`) — всё это Opus в контейнере Ogg.
Отправляйте как есть: сервер определяет контейнер по байтам, поэтому
расширение файла роли не играет:

```sh
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" --data-binary @voice.ogg

gigastt transcribe voice.opus
```

Проверка: HTTP 200 с транскриптом. Opus декодируется в свои родные 48 кГц
и ресемплируется внутри; поддерживаются моно и стерео (стерео миксуется
в моно, если не включить разделение каналов ниже), multistream (>2 каналов)
OGG/Opus отклоняется.

### Стерео-звонок → два спикера (channels=split)

Многие АТС пишут каждого участника в свой канал: левый — один спикер, правый —
другой. Разделение каналов транскрибирует каждый канал как отдельного спикера
вместо микширования в моно: канал 0 (левый) становится `speaker_0`, канал 1
(правый) — `speaker_1`.

```sh
# REST
curl -X POST "http://127.0.0.1:9876/v1/transcribe?channels=split" \
  -H "Content-Type: application/octet-stream" --data-binary @call.wav

# CLI — SRT с метками спикеров [SPEAKER_0] / [SPEAKER_1] в репликах
gigastt transcribe call.wav --stereo-speakers -f srt -o call.srt
```

В JSON каждое слово получает поле `speaker`, слова упорядочены по времени
начала:

```json
{
  "text": "…",
  "words": [
    {"word": "покажи", "start": 0.08, "end": 0.48, "confidence": 0.95, "speaker": 1},
    {"word": "шестьдесят", "start": 0.52, "end": 1.08, "confidence": 0.96, "speaker": 0}
  ],
  "duration": 3.43
}
```

Для отчётов группируйте слова по `speaker` — получите текст и время речи
каждой стороны (агент vs клиент). Какая сторона в каком канале — конвенция
вашей АТС; откалибруйтесь один раз на звонке с известными спикерами.

Проверка: присутствуют обе метки —

```sh
curl -s -X POST "http://127.0.0.1:9876/v1/transcribe?channels=split" \
  -H "Content-Type: application/octet-stream" --data-binary @call.wav \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(sorted({w.get('speaker') for w in d['words']}))"
# [0, 1]
```

Откаты, о которых стоит знать: если файл моно, каналов больше двух или
запись dual-mono (каналы почти идентичны — некоторые АТС пишут микс звонка
в оба канала), gigastt откатывается на обычный моно-транскрипт без полей
`speaker` и пишет в лог предупреждение `falling back to mono transcription`.
Разделение каналов несовместимо с диаризацией: `channels=split&diarization=true`
возвращает `400 conflicting_modes`.

### RTP-дамп без контейнера

RTP-захват, очищенный до полезной нагрузки, не имеет заголовка для сниффинга,
поэтому кодек нужно объявить. Экспортируйте только payload из вашего
инструмента (в Wireshark: Telephony → RTP → Stream Analysis → сохранение
payload; у SBC есть похожий экспорт) — 12-байтовых RTP-заголовков в файле
быть не должно.

```sh
# Захват G.711 A-law на 8 кГц
curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=pcma&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" --data-binary @dump.alaw

# Захват G.722 — см. замечание про clock-rate в SDP ниже
curl -X POST "http://127.0.0.1:9876/v1/transcribe?codec=g722&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" --data-binary @dump.g722

# Эквивалент для CLI
gigastt transcribe dump.g722 --codec g722 --sample-rate 8000
```

Особенность G.722 в SDP: по историческим причинам SDP/RTP анонсирует G.722
с clock-rate 8000 Гц, хотя поток на самом деле декодируется в 16 кГц. gigastt
принимает для `g722` и `8000`, и `16000` и всегда декодирует в 16 кГц, так что
подойдёт любое значение. Для сырого G.711 (`pcmu`/`pcma`) принимается любая
частота в диапазоне 8000–48000 Гц, с ресемплингом.

Проверка: HTTP 200 и осмысленный текст. Ошибки параметров срабатывают сразу,
до инференса: `codec` без `sample_rate` → `400 invalid_sample_rate`
(«sample_rate is required when codec is set»); неизвестный кодек →
`400 unsupported_codec` («Unsupported codec. Supported: pcmu (ulaw),
pcma (alaw), g722»).

Потери в захвате: поток декодируется как есть. Дыры от потерянных пакетов
и переупорядоченное аудио не восстанавливаются — при дырах в транскрипте
подавите джиттер или переснимите захват.

### Папка записей (пакетная обработка)

`gigastt transcribe-batch` сканирует файлы с расширениями `wav`, `mp3`, `m4a`,
`ogg` и `flac` — это покрывает G.711/G.722 WAV и OGG/Opus (`.ogg`) из
коробки:

```sh
gigastt transcribe-batch recordings/ out/ --format txt,json
```

Два вида телефонных файлов **не** сканируются:

- **файлы `.opus`** — переименуйте их в `.ogg`. Внутри контейнер Ogg, он
  определяется по байтам, поэтому переименование безопасно:

  ```sh
  for f in recordings/*.opus; do mv "$f" "${f%.opus}.ogg"; done
  ```

- **сырые потоки `.ulaw` / `.alaw` / `.g722`** — сначала оберните в WAV
  (у batch нет флага `--codec`), затем запустите batch на обёрнутых файлах:

  ```sh
  mkdir -p wav
  for f in recordings/*.ulaw; do
    ffmpeg -y -v error -f mulaw -ar 8000 -ac 1 -i "$f" \
      -ar 16000 -ac 1 -c:a pcm_s16le "wav/$(basename "${f%.ulaw}").wav"
  done
  gigastt transcribe-batch wav/ out/ --format txt,json
  ```

  Для A-law используйте `-f alaw`, для G.722 — `-f g722` (G.722 не нужен
  входной `-ar`).

Проверка: в `out/` по одному `.txt`/`.json` на каждый исходный файл, команда
завершается с кодом 0. Если во входящую папку постоянно падают новые записи,
то же самое в непрерывном режиме делает `gigastt watch` — подробности о
batch/watch и длинных записях в главе
[CLI и пакетная обработка](02-cli-batch.md).

## Шпаргалка по форматам

| Вход | Определяется по контейнеру | Нужен `?codec=` / `--codec` | Замечания и ограничения |
|---|---|---|---|
| WAV PCM (8–32 бита, IEEE float) | да | нет | стерео автоматически миксуется в моно |
| WAV с G.711 A-law / μ-law | да | нет | обычно 8 кГц, ресемплинг в 16 кГц |
| WAV с G.722 ADPCM (теги `0x0064`, `0x028F`) | да | нет | декодируется в родные 16 кГц |
| OGG/Opus, `.opus` | да | нет | только моно/стерео; >2 каналов отклоняется |
| сырой `.ulaw` / `.alaw` | нет (без заголовков) | да — `pcmu` / `pcma` | `sample_rate` 8000–48000 |
| сырой `.g722` | нет (без заголовков) | да — `g722` | `sample_rate` 8000 (конвенция SDP) или 16000; декодируется в 16 кГц |
| wav49 (GSM 06.10 в WAV) | да | н/д | не декодируется — сначала конвертировать в PCM16 WAV |
| G.729 (любая обёртка) | да | н/д | не поддерживается — сначала конвертировать в PCM16 WAV |

Общее для всех путей: загрузка ограничена 30 минутами декодированного аудио,
а `?codec=` / `?sample_rate=` одинаково работают на `/v1/transcribe`,
`/v1/transcribe/stream` и `/v1/jobs`. Deepgram-совместимый эндпоинт
`/v1/listen`, принимающий те же телефонные форматы, в работе.

## Проверка результата

В репозитории лежат телефонные фикстуры, которые используют его собственные
e2e-тесты, — 4 секунды русской речи («шестьдесят тысяч тенге сколько будет
стоить»), перекодированные во все поддерживаемые телефонные форматы. Нужны
скачанная модель и запущенный сервер:

```sh
# сырой путь
curl -s -X POST "http://127.0.0.1:9876/v1/transcribe?codec=pcmu&sample_rate=8000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @crates/gigastt/tests/fixtures/telephony/speech.ulaw

# контейнерный путь — без параметров
curl -s -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @crates/gigastt/tests/fixtures/telephony/speech_g722.wav
```

Оба запроса возвращают HTTP 200 с транскриптом, где есть «тенге» и «стоить», —
при включённых на сервере пунктуации и ITN текст выглядит как «60000 тенге,
сколько будет стоить?». Если ваш файл падает, а эти фикстуры проходят,
проблема в файле, а не в сервере — возвращайтесь к шагу 0.

## Частые ошибки

- **`422` — «Check audio format»** (`invalid_audio` / `transcription_error`).
  Байты были проверены как контейнер, и декодирование не удалось. Обычные
  причины: сырой поток отправлен без `codec=`; файл wav49 (GSM) или G.729;
  обрезанная/битая загрузка. Определите файл шагом 0.
- **G.729 не поддерживается.** Сырая загрузка с `?codec=g729` возвращает
  `400 unsupported_codec`; G.729-в-WAV падает с `422`. Конвертируйте ffmpeg
  и отправляйте результат:

  ```sh
  ffmpeg -y -i call_g729.wav -ar 16000 -ac 1 -c:a pcm_s16le call_16k.wav
  ```

- **`?codec=` на файле-контейнере.** Параметр полностью переопределяет
  сниффинг: WAV, отправленный с `?codec=pcmu`, декодирует WAV-заголовок как
  μ-law-шум и возвращает 200 с мусором. Используйте `codec=` только для
  потоков без заголовков.
- **RTP-дамп с заголовками или джиттером.** `?codec=` ждёт только байты
  payload. Оставшиеся 12-байтовые RTP-заголовки декодируются как периодические
  щелчки, а дыры от потерь и переупорядочивания превращаются в провалы
  транскрипта — на сервере ничего не восстанавливается. Экспортируйте только
  payload и подавите джиттер до загрузки.
- **«8-битный WAV» — это не поломка.** G.711 в WAV законно показывает
  `bits_per_sample=8` (компандированные сэмплы) — отправляйте как есть.
  Настоящая ловушка обратная: при собственной конвертации всегда пишите
  16-битный PCM (`pcm_s16le`); 8-битный линейный PCM (`pcm_u8`) убивает
  точность распознавания.
- **Путаница моно/стерео.** `channels=split` на моно-файле, файле с >2
  каналами или dual-mono стерео (некоторые АТС пишут микс звонка в оба
  канала) молча откатывается на обычный моно-транскрипт — без полей
  `speaker`, лишь с предупреждением `falling back to mono transcription`
  в логе. Если АТС пишет сведённое моно, никакой флаг не разделит спикеров
  задним числом; пишите стерео или используйте `diarization=true`
  (несовместимо с `channels=split`).
- **«Audio file too long»**. Одна загрузка ограничена 30 минутами
  декодированного аудио («Maximum supported: 1800s»). Делите длинные записи —
  например, по плечам звонка — перед загрузкой.
- **Перепутаны A-law/μ-law.** Оба закона декодируются «успешно», поэтому
  неверный выбор возвращает 200 с мусором вместо ошибки. Если сырой поток
  транскрибируется в шум, повторите с другим именем кодека.

## Ссылки

- [CLI и пакетная обработка](02-cli-batch.md) — пакетный и watch-режимы,
  форматы экспорта, длинные записи.
- [Начало работы](01-getting-started.md) — установка и скачивание модели.
- [Введение](README.md) — карта документации.
- [docs/api.md — Audio formats and telephony codecs](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#audio-formats-and-telephony-codecs) —
  каноническая таблица форматов и все query-параметры.
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) —
  справочник флагов `transcribe`, `transcribe-batch` и `watch`.
- [crates/gigastt-core/src/inference/audio.rs](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-core/src/inference/audio.rs) —
  внутреннее устройство декодеров (сниффинг G.722, путь Opus, детект
  dual-mono).
- [Телефонные и Opus-фикстуры](https://github.com/ekhodzitsky/gigastt/tree/main/crates/gigastt/tests/fixtures)
  с их генераторами
  [generate_telephony_fixtures.sh](https://github.com/ekhodzitsky/gigastt/blob/main/scripts/generate_telephony_fixtures.sh)
  и
  [generate_opus_fixtures.sh](https://github.com/ekhodzitsky/gigastt/blob/main/scripts/generate_opus_fixtures.sh).
