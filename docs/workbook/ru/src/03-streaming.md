# Стриминг по WebSocket

Живая транскрипция с промежуточными результатами: микрофон, телефонная
линия или захват звука в браузере поступают как сырой PCM16, а текст
появляется примерно через секунду после речи. Эта глава — книга рецептов
для такой интеграции. Пофилдовый справочник протокола остаётся в
[docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#websocket--real-time-streaming),
машиночитаемая схема — в
[docs/asyncapi.yaml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/asyncapi.yaml);
мы ссылаемся на них, а не повторяем.

## Сценарий

Вы строите real-time интеграцию — живые субтитры для встреч, голосового
бота на телефонной линии или «диктофон с мгновенным текстом». Аудио идёт
непрерывно; пользователи ждут, что транскрипт растёт прямо во время речи,
каждая фраза финализируется чисто, а хвостовые слова не теряются при
завершении потока. Некоторые сессии длятся часами, в некоторых конфигурациях
захватываются два источника сразу (микрофон + системный звук), а клиент
обязан переживать насыщение пула и сетевые обрывы без ручного вмешательства.

## Требования

- Запущенный сервер с моделью — по главе [Начало работы](01-getting-started.md):
  `gigastt serve` (первый запуск скачивает ~850 МБ и квантизирует — ждите
  готовности, не убивайте процесс):

  ```sh
  until curl -sf http://127.0.0.1:9876/ready; do sleep 2; done
  # {"status":"ready","pool_available":2,"pool_total":2}
  ```

- WebSocket-стек для вашего языка: `pip install websockets` для рецептов на
  Python, Node.js ≥ 22 (глобальный `WebSocket`, без зависимостей) для
  JavaScript, Go 1.23+ для SDK.
- Источник PCM16 mono. Для копипастных проверок ниже в репозитории есть
  4-секундная фикстура русской речи ровно в нужном формате (16 кГц mono
  Int16): `crates/gigastt/tests/fixtures/golos_00.wav`. Для реального
  микрофона `ffmpeg` превращает любое устройство захвата в сырой PCM16 на
  stdout.

## Рецепт

Форма сессии, на которую опираются все рецепты, — подключение, чтение
`ready`, опциональный `configure`, поток бинарных PCM16, `stop` для
финализации:

```
Client                            Server
  |-------- connect --------------> |
  | <------- ready ----------------- |  лимиты и supported rates (читайте их!)
  |------- configure (optional) --> |  до первого аудиофрейма
  |-------- binary PCM16 ---------> |
  | <------- partial / final ------ |
  |--------- stop ----------------> |
  | <------- final ----------------- |  хвостовые слова сброшены, затем close
```

### Рецепт 1: подключаемся, согласуем параметры и стримим с микрофона

1. **Подключитесь и сначала прочитайте `ready`.** Первое сообщение сервера
   — всегда `ready`. В нём всё, что нельзя хардкодить: `version` протокола,
   допустимые `supported_rates` и лимиты сессии (`max_session_secs`,
   `idle_timeout_secs`), вокруг которых строится рецепт 4:

   ```json
   {
     "type": "ready",
     "model": "gigaam-v3-rnnt",
     "sample_rate": 48000,
     "version": "1.0",
     "supported_rates": [8000, 16000, 24000, 44100, 48000],
     "max_session_secs": 3600,
     "idle_timeout_secs": 300
   }
   ```

   `sample_rate` (по умолчанию 48000) — это то, что сервер ждёт, если вы не
   пришлёте `configure`. Одна деталь, которую стоит знать до отладки
   «молчащего подключения»: при насыщенном пуле сервер отвечает сообщением
   `error` *вместо* `ready` — см. рецепт 5.

2. **Отправьте `configure` до первого аудиофрейма.** Выберите частоту,
   которую реально выдаёт ваш конвейер захвата, — она обязана быть из
   `ready.supported_rates`. 16 кГц — оптимум, когда источник под вашим
   контролем (модель внутри работает на 16 кГц, ресемплинг не нужен);
   браузерный захват на 48 кГц тоже можно слать как есть. Неподдерживаемая
   частота не фатальна: вы получите ошибку `invalid_sample_rate`, а сессия
   продолжится на прежней частоте. `configure`, пришедший после первого
   аудиофрейма, отклоняется с `configure_too_late`, настройки сохраняются —
   поэтому шлите его сразу после `ready`:

   ```json
   {"type": "configure", "sample_rate": 16000}
   ```

3. **Шлите бинарные фреймы PCM16** (signed 16-bit little-endian, mono) на
   согласованной частоте. Разумная нарезка: **100–500 мс на фрейм**
   (3 200–16 000 байт при 16 кГц). Партилы формируются со шагом декодирования
   примерно 0.8 с нового аудио, поэтому субсекундные чанки сохраняют
   «живость» превью без лишних накладных расходов. Жёсткий потолок —
   `--ws-frame-max-bytes` (по умолчанию 512 КиБ): больший фрейм закрывает
   сокет с кодом 1009. Нечётная длина допустима (последний байт
   переносится в следующий фрейм), случайные пустые фреймы терпимы.
   Стереоисточники сначала микшируйте в mono (`-ac 1` в ffmpeg).

4. **Собираем всё вместе — микрофон в живой текст.** Конвейер гонит
   микрофон через ffmpeg в минимальный Python-клиент:

   ```sh
   # macOS (-f avfoundation); Linux: -f alsa -i default; Windows: -f dshow -i audio="..."
   ffmpeg -hide_banner -loglevel error -f avfoundation -i ":default" \
     -ac 1 -ar 16000 -f s16le - | python3 mic_stream.py
   ```

   `mic_stream.py` — полный эталонный цикл, который использует эта глава:

   ```python
   #!/usr/bin/env python3
   """Stream raw PCM16 from stdin to gigastt and print live transcripts.

   Usage: ffmpeg ... -f s16le - | python3 mic_stream.py [label] [server]
   """
   import asyncio
   import json
   import sys

   import websockets

   LABEL = sys.argv[1] if len(sys.argv) > 1 else "mic"
   SERVER = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9876/v1/ws"
   RATE = 16000
   CHUNK = RATE * 2 // 5  # 400 ms of PCM16


   async def main() -> None:
       async with websockets.connect(SERVER) as ws:
           ready = json.loads(await ws.recv())
           assert ready["type"] == "ready", ready
           assert RATE in ready.get("supported_rates", [ready["sample_rate"]])
           await ws.send(json.dumps({"type": "configure", "sample_rate": RATE}))
           print(f"{LABEL}: connected to {ready['model']}", file=sys.stderr)

           async def receive() -> None:
               async for raw in ws:
                   msg = json.loads(raw)
                   if msg["type"] == "partial":
                       print(f"\r{LABEL} ... {msg['text']}   ", end="", flush=True)
                   elif msg["type"] == "final":
                       conf = msg.get("confidence")
                       suffix = f" ({conf:.2f})" if conf is not None else ""
                       print(f"\r{LABEL} >>> {msg['text']}{suffix}   ")
                   elif msg["type"] == "error":
                       print(f"\n{LABEL} ERR {msg['code']}: {msg['message']}")

           receiver = asyncio.create_task(receive())
           try:
               while data := await asyncio.to_thread(sys.stdin.buffer.read, CHUNK):
                   await ws.send(data)
               # Capture ended: finalize — never close before the trailing final.
               await ws.send(json.dumps({"type": "stop"}))
               await receiver
           except websockets.ConnectionClosed:
               pass  # server closed after the trailing final (or an error we printed)


   asyncio.run(main())
   ```

   Блокирующее чтение stdin выполняется в потоке (`asyncio.to_thread`), чтобы
   задача-приёмник продолжала печатать, пока вы говорите.

**Проверка:** во время речи строки `...` обновляются примерно раз в
секунду; пауза около полусекунды даёт `>>>` final. Закрытие ffmpeg (Ctrl+C)
вызывает последний `final` и чистое закрытие — ничего не обрезается.

### Рецепт 2: partial показываем вживую, final — в транскрипт

Оба типа сообщений несут одинаковую полезную нагрузку, но играют разные
роли в UI — относитесь к ним по-разному.

- **`partial` — это превью.** Всегда сырая гипотеза декодера: нижний
  регистр, без пунктуации, и она *может измениться* с приходом нового
  аудио. Новый partial появляется примерно каждые 0.8 с речи (шаг
  декодирования):

  ```json
  {
    "type": "partial",
    "text": "привет как",
    "timestamp": 1712700000.123,
    "is_final": false,
    "confidence": 0.93,
    "words": [
      {"word": "привет", "start": 0.0, "end": 0.4, "confidence": 0.97},
      {"word": "как", "start": 0.5, "end": 0.7, "confidence": 0.89}
    ]
  }
  ```

- **`final` — это зафиксированная строка.** Обогащается на границе
  финализации: инверсная нормализация текста (числительные → цифры), затем
  восстановление пунктуации и регистра. Голове `rnnt` нужна подключённая
  пунктуационная модель (серверный `--punctuation auto` по умолчанию;
  проверка: `GET /health` → `"punctuation":true`); голова `e2e_rnnt`
  пунктуирует сама — см. [Модели и бэкенды](04-models-and-backends.md).
  `words[]` в обоих типах сообщений всегда хранят сырой вывод декодера;
  переписывается только склеенный `text`:

  ```json
  {
    "type": "final",
    "text": "Привет, как дела?",
    "timestamp": 1712700001.456,
    "is_final": true,
    "confidence": 0.95,
    "words": [
      {"word": "привет", "start": 0.0, "end": 0.4, "confidence": 0.97},
      {"word": "как", "start": 0.5, "end": 0.7, "confidence": 0.93},
      {"word": "дела", "start": 0.8, "end": 1.1, "confidence": 0.95}
    ]
  }
  ```

Правило для UI: последний `partial` рисуем в стиле «превью» (серым/курсивом,
заменяемым), а когда приходит `final` для этой фразы — заменяем превью на
`final.text` и дописываем в зафиксированный транскрипт. Текст partial
никогда не сохраняем.

`final` срабатывает при завершении фразы: встроенный эндпоинтинг ловит
примерно 0.6 с затихания, а с опциональным Silero VAD (серверный `--vad`,
модель скачивается автоматически) конец фразы следует за
`--vad-min-silence-ms` (по умолчанию 500 мс). Непрерывная речь без пауз
тоже финализируется — стриминговое окно упирается в ~2.5 с и сдвигается,
так что транскрипт коммитится даже сквозь монолог. `final` также приходит
по `stop` (рецепт 3) и перед закрытием по инициативе сервера (рецепт 4).

Сессионные переопределения постобработки комбинируются тем же сообщением
`configure` (только для финалов; partial всегда сырые). Запрос
`punctuation: true` на сервере без модели — мягкий no-op, а не ошибка:

```json
{"type": "configure", "sample_rate": 16000, "punctuation": false, "itn": false}
```

**Проверка:** скажите «привет как дела» с короткими паузами. Превью
показывает `привет как дела` в нижнем регистре; зафиксированная строка
приходит как `Привет, как дела?` (когда пунктуационная модель подключена —
проверьте `curl -s http://127.0.0.1:9876/health` с `"punctuation":true`).

### Рецепт 3: корректное завершение — `stop` вместо drain

Единственно правильный паттерн конца потока:

1. Отправьте `{"type": "stop"}`.
2. **Дождитесь `final`** — сервер декодирует всё, что ещё буферизовано с
   последнего partial (хвостовые слова не теряются), и выдаёт последний
   `final`, возможно с пустым `text`, если ничего не оставалось.
3. Только потом закрывайте сокет (или позвольте серверу закрыть его — он
   завершает сессию сразу после этого final).

**Не** закрывайте сокет сразу после последнего аудиофрейма (хвост короче
шага декодирования будет потерян) и **не** вставляйте фиксированный `sleep`
для drain — `final` после `stop` и есть явный, безпотерный маркер конца.
Цикл `mic_stream.py` из рецепта 1 уже реализует этот паттерн.

Keepalive не требует кода с вашей стороны: сервер шлёт ping каждые 30 с и
закрывает соединение после двух подряд ping без входящих фреймов между ними
(≈ 90 с на обнаружение полуоткрытого пира). Любой входящий фрейм — pong,
бинарное аудио или текст — сбрасывает счётчик, а стандартные
WebSocket-клиенты отвечают на ping автоматически. Отвечать на ping вручную
нужно только в реализациях на голых сокетах.

**Проверка:** скажите одно слово и немедленно отправьте `stop`, до прихода
любого partial. Слово всё равно появится в хвостовом `final`. Затем
посмотрите на закрытие: оно происходит *после* этого final, а не до.

### Рецепт 4: держим длинные сессии (idle-таймаут и потолок сессии)

Каждую сессию ограничивают два серверных таймера; оба объявляются в
`ready`, чтобы вы планировали заранее, а не узнавали о них из close-фрейма.

- **Idle-таймаут** (`idle_timeout_secs`, по умолчанию 300): любой фрейм —
  аудио, pong или текст — сбрасывает его. Паузы в речи считаются простоем,
  только если клиент перестаёт слать; **поток тихого PCM держит сессию
  живой (тишина — тоже аудио)**. При срабатывании: ошибка `idle_timeout` и
  close 1001. Если ваше приложение глушит захват на паузах, шлите фреймы
  тишины вместо ничего.
- **Потолок сессии** (`max_session_secs`, по умолчанию 3600, `0` —
  отключён): настенные часы от момента подключения. При срабатывании сервер
  шлёт ошибку `max_session_duration_exceeded`, **сначала сбрасывает
  `final`**, затем закрывает с 1008 — всё уже распознанное сохранено,
  поэтому переподключение безопасно.

Для записей длиннее часа есть два варианта, и их можно комбинировать:

1. **Сторона оператора:** поднять или отключить потолок — `gigastt serve
   --max-session-secs 0` (каждый лимит — CLI-флаг; см.
   [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)).
2. **Сторона клиента:** ротация по расписанию. Прочитайте потолок из
   `ready` и на ~90 % от него финализируйте `stop` и откройте свежую
   сессию — плановый реконнект, а не обрыв:

   ```python
   # rotate_sessions.py — keep transcription running across the server cap.
   import asyncio
   import json

   import websockets

   SERVER = "ws://127.0.0.1:9876/v1/ws"


   def handle_message(msg: dict) -> None:
       if msg["type"] == "final":
           print(">>>", msg["text"])
       elif msg["type"] == "error":
           print("ERR", msg["code"], msg["message"])


   async def run_session(ws, stream_audio, rotate_at: float | None) -> None:
       async def rotate() -> None:
           await asyncio.sleep(rotate_at)
           await ws.send(json.dumps({"type": "stop"}))  # flush now, reconnect fresh

       tasks = [asyncio.create_task(stream_audio(ws))]
       if rotate_at:
           tasks.append(asyncio.create_task(rotate()))
       try:
           async for raw in ws:  # ends when the server closes after the final
               handle_message(json.loads(raw))
       finally:
           for task in tasks:
               task.cancel()


   async def run_shift(stream_audio) -> None:
       """Rotate before the cap; back off on transient drops."""
       backoff = 0.25
       while True:
           try:
               async with websockets.connect(SERVER) as ws:
                   ready = json.loads(await ws.recv())
                   cap = ready["max_session_secs"]  # always sent; 0 = no cap
                   await run_session(ws, stream_audio, rotate_at=cap * 0.9 or None)
                   backoff = 0.25  # clean stop → final → close: rotate at once
           except (OSError, websockets.ConnectionClosed):
               # 1006 drop, a proxy-injected 1011, the 1008 cap close, network...
               await asyncio.sleep(backoff)
               backoff = min(backoff * 2, 30)
   ```

   `stream_audio(ws)` — ваш цикл захвата из рецепта 1. Обратите внимание,
   чего этот паттерн *не* делает: он никогда не ретраит фатальные ошибки
   «чините-клиента» (`unsupported_protocol_version`, частота вне
   `supported_rates`) — те падают сразу на рукопожатии и должны чиниться, а
   не ретраиться.

**Проверка:** запустите сервер с крошечным потолком, чтобы увидеть весь
жизненный цикл за минуту: `gigastt serve --max-session-secs 30
--idle-timeout-secs 10`. С идущим аудио вы увидите ошибку
`max_session_duration_exceeded`, сброшенный `final`, close 1008 и
переподключение ротатора без потери текста. Полностью прекратите слать
фреймы — через 10 с сработает ошибка `idle_timeout` с close 1001.

### Рецепт 5: пороги confidence и обратное давление

**Confidence.** Каждый сегмент транскрипта несёт опциональный `confidence` —
взвешенное по длительности среднее его `words[].confidence` (у слова это
средний softmax по его BPE-токенам). Это среднее softmax-оценок, **не
калиброванная вероятность**, и оно опускается, когда в сегменте нет слов.
Стартовые пороги для подстройки на ваших данных: подсвечивайте для
человеческой проверки слова ниже ~0.7, помечайте сегменты ниже ~0.8:

```js
const unsure = (msg.words ?? []).filter((w) => w.confidence < 0.7);
if (unsure.length) console.log("review:", unsure.map((w) => w.word).join(" "));
```

**Обратное давление.** Слоты инференса берутся из пула (`--pool-size`, по
умолчанию 2), и WebSocket-сессия держит свой слот всё своё время жизни.
Когда все слоты заняты, новое подключение ждёт до 30 с — и если слот не
освобождается, сервер отвечает *вместо `ready`* ошибкой `timeout` с
`retry_after_ms`, затем закрывает. Точно соблюдайте подсказку вместо
угадывания задержки:

```json
{"type": "error", "message": "Server busy, try again later", "code": "timeout", "retry_after_ms": 30000}
```

```python
first = json.loads(await ws.recv())          # may be an error, not ready
if first["type"] == "error" and first["code"] == "timeout":
    await asyncio.sleep(first["retry_after_ms"] / 1000)  # then reconnect
```

Для всех остальных неожиданных закрытий сокета — обрыв 1006, код вроде
1011, подставленный прокси, сетевой сбой — переподключайтесь с
экспоненциальной задержкой (старт ~250 мс, удвоение до нескольких секунд,
потолок попыток), как в `run_shift` из рецепта 4. Фатальные ошибки
рукопожатия (`unsupported_protocol_version`) не ретрайте. Оба официальных
SDK реализуют ровно эту политику — configure-first рукопожатие,
`retry_after_ms` при насыщении пула, экспоненциальная задержка в остальных
случаях — поэтому предпочитайте их самодельным велосипедам:
[sdks/go](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/go),
[sdks/js](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/js).

**Проверка:** запустите с `--pool-size 1` и подключите два клиента
одновременно. Второй ждёт ~30 с, затем получает
`{"code":"timeout","retry_after_ms":30000}`; выспавшись положенное, он
подключается после завершения первой сессии. Для confidence: прогоните
чистый речевой файл и убедитесь, что финалы несут `confidence` около 1.0,
затем попробуйте тихое/шумное аудио и понаблюдайте, как слова проваливаются
ниже 0.7.

### Рецепт 6: два канала — микрофон + системный звук

Паттерн ассистента встреч (ваш микрофон + системный звук собеседника,
каждый со своей меткой) раскладывается на **две независимые
WebSocket-сессии** — сервер не помечает источники, поэтому метки ставятся на
клиенте. Ограничение, вокруг которого надо проектировать, — пул: каждая
сессия держит один инференс-слот всё своё время жизни, поэтому дефолтный
`--pool-size 2` вмещает ровно два канала и больше ничего. Дайте серверу
запас, если подключаются и другие клиенты, — каждый дополнительный слот
стоит примерно 0.4 ГБ RAM с INT8-энкодером (сервер ограничивает пул по
доступной памяти при загрузке):

```sh
gigastt serve --pool-size 4
```

Затем запустите по конвейеру захвата на источник, каждый со своей меткой
(`mic_stream.py` из рецепта 1 принимает метку первым аргументом):

```sh
# terminal 1 — your microphone (macOS example; Linux: -f alsa -i default)
ffmpeg -hide_banner -loglevel error -f avfoundation -i ":default" \
  -ac 1 -ar 16000 -f s16le - | python3 mic_stream.py mic

# terminal 2 — system audio (Linux Pulse/PipeWire monitor source;
# macOS: a virtual device such as BlackHole selected via avfoundation)
ffmpeg -hide_banner -loglevel error -f pulse -i default.monitor \
  -ac 1 -ar 16000 -f s16le - | python3 mic_stream.py system
```

Две более лёгкие альтернативы, когда помеченные каналы не нужны: смикшируйте
оба источника в один поток на клиенте (один слот пула, одна сессия) или
возьмите пословные метки говорящих из диаризации — сервер, собранный с
`--features diarization`, объявляет `"diarization": true` в `ready`, и
`{"type":"configure","diarization":true}` добавляет поле `speaker` к каждому
слову (см. справочник в
[docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md)).

**Проверка:** оба терминала печатают со своими метками `mic` / `system`;
пока оба работают, `curl -s http://127.0.0.1:9876/ready` показывает
`"pool_available":0,"pool_total":2` (с дефолтным пулом), а третий клиент
получает обработку `timeout` + `retry_after_ms` из рецепта 5.

### Рецепт 7: клиентские скелеты (Python, Node.js, Go)

Минимальные, но полные циклы для старта — каждый делает полный оборот
ready → configure → stream → stop → wait-for-final → close с обработкой
ошибок. Больше клиентов (Bun, Kotlin, Rust) лежит в
[examples/](https://github.com/ekhodzitsky/gigastt/tree/main/examples).

**Python** — стримит WAV 16 кГц mono PCM16, соблюдает `retry_after_ms` при
насыщении пула:

```python
#!/usr/bin/env python3
"""Stream a 16 kHz mono PCM16 WAV to gigastt, honoring server backpressure.

Usage: python3 stream_wav.py <audio.wav> [ws://host:port]
"""
import asyncio
import json
import sys
import wave

import websockets


class PoolBusy(Exception):
    """Pool saturated: the server sent retry_after_ms instead of ready."""


async def session(path: str, server: str) -> None:
    async with websockets.connect(server) as ws:
        first = json.loads(await ws.recv())
        if first["type"] == "error":  # pool busy: error + close instead of ready
            raise PoolBusy(first.get("retry_after_ms", 30000))
        assert first["type"] == "ready", first
        await ws.send(json.dumps({"type": "configure", "sample_rate": 16000}))

        with wave.open(path, "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2, "mono PCM16 WAV"
            pcm = wav.readframes(wav.getnframes())

        async def receive() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "partial":
                    print(f"\r... {msg['text']}   ", end="", flush=True)
                elif msg["type"] == "final":
                    print(f"\r>>> {msg['text']}   ")
                elif msg["type"] == "error":
                    print(f"\nERR {msg['code']}: {msg['message']}")

        receiver = asyncio.create_task(receive())
        try:
            for off in range(0, len(pcm), 16000):  # 0.5 s of PCM16 at 16 kHz
                await ws.send(pcm[off : off + 16000])
                await asyncio.sleep(0.1)  # pace it like a live feed
            await ws.send(json.dumps({"type": "stop"}))
            await receiver  # trailing final, then the server closes
        except websockets.ConnectionClosed:
            pass


async def main() -> None:
    server = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9876/v1/ws"
    while True:
        try:
            await session(sys.argv[1], server)
            return
        except PoolBusy as busy:
            wait = busy.args[0] / 1000
            print(f"pool busy, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)  # honor retry_after_ms exactly


asyncio.run(main())
```

**Node.js** — без зависимостей (глобальный `WebSocket`, Node ≥ 22), тот же
контракт WAV-файла:

```js
// stream_wav.mjs — stream a 16 kHz mono PCM16 WAV to gigastt (Node.js ≥ 22).
import { readFile } from "node:fs/promises";

const [wavPath, server = "ws://127.0.0.1:9876/v1/ws"] = process.argv.slice(2);
if (!wavPath) {
  console.error("usage: node stream_wav.mjs <audio.wav> [ws://host:port]");
  process.exit(1);
}

const pcm = (await readFile(wavPath)).subarray(44); // skip the WAV header
const ws = new WebSocket(server);
ws.binaryType = "arraybuffer";

let stopped = false;
const done = new Promise((resolve, reject) => {
  ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "ready") {
      ws.send(JSON.stringify({ type: "configure", sample_rate: 16000 }));
      for (let off = 0; off < pcm.byteLength; off += 16000) { // 0.5 s at 16 kHz
        ws.send(pcm.subarray(off, off + 16000));
        await new Promise((r) => setTimeout(r, 100)); // pace like a live feed
      }
      stopped = true;
      ws.send(JSON.stringify({ type: "stop" })); // finalize — do NOT close yet
    } else if (msg.type === "partial") {
      process.stdout.write(`\r... ${msg.text}   `);
    } else if (msg.type === "final") {
      console.log(`\r>>> ${msg.text}   `);
      if (stopped) { ws.close(); resolve(); } // trailing final: safe to close
    } else if (msg.type === "error") {
      const hint = msg.retry_after_ms ? ` (retry in ${msg.retry_after_ms} ms)` : "";
      reject(new Error(`${msg.code}: ${msg.message}${hint}`));
    }
  };
  ws.onerror = () => reject(new Error("websocket transport error"));
});
await done;
```

Для всего серьёзнее скелета типизированный SDK из коробки добавляет
политику реконнекта из рецепта 5: `npm install @gigastt/client` — см.
[sdks/js](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/js).

**Go** — на официальном SDK (`go get github.com/ekhodzitsky/gigastt/sdks/go@latest`),
который пинит версию протокола, шлёт `configure` первым и ретраит обратное
давление с серверным `retry_after_ms`:

```go
// stream_wav.go — stream a 16 kHz mono PCM16 WAV to gigastt via the Go SDK.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	gigastt "github.com/ekhodzitsky/gigastt/sdks/go"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run stream_wav.go <audio.wav>")
	}
	done := make(chan struct{})

	client, err := gigastt.Dial(context.Background(), gigastt.DefaultURL,
		gigastt.WithSampleRate(16000), // must be in the server's supported_rates
		gigastt.WithReconnect(250*time.Millisecond, 5*time.Second, 10),
		gigastt.WithHandlers(gigastt.Handlers{
			OnPartial: func(t gigastt.Transcript) { fmt.Printf("\r... %s   ", t.Text) },
			OnFinal:   func(t gigastt.Transcript) { fmt.Printf("\r>>> %s   \n", t.Text) },
			OnError:   func(e *gigastt.ServerError) { log.Printf("server error: %v", e) },
			OnClose: func(err error) {
				if err != nil {
					log.Printf("connection closed: %v", err)
				}
				close(done)
			},
		}),
	)
	if err != nil {
		log.Fatal(err) // e.g. *gigastt.ServerError unsupported_protocol_version
	}
	defer client.Close()

	wav, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	for off := 44; off < len(wav); off += 16000 { // skip WAV header; 0.5 s chunks
		if err := client.SendPCM(wav[off:min(off+16000, len(wav))]); err != nil {
			log.Fatal(err) // ErrReconnecting: drop or retry the frame yourself
		}
		time.Sleep(100 * time.Millisecond) // pace like a live feed
	}
	if err := client.Stop(); err != nil { // finalize; server closes after the final
		log.Fatal(err)
	}
	<-done
}
```

**Проверка:** направьте любой из трёх на фикстуру из репозитория —
`python3 stream_wav.py crates/gigastt/tests/fixtures/golos_00.wav` — и
получите несколько `...` partial, за которыми следуют `>>>` final русской
речи, а затем чистый выход после хвостового final.

## Проверка результата

Сквозная проверка в двух терминалах:

```sh
# terminal 1
gigastt serve

# terminal 2 — wait for readiness, then stream the bundled speech fixture
until curl -sf http://127.0.0.1:9876/ready; do sleep 2; done
python3 stream_wav.py crates/gigastt/tests/fixtures/golos_00.wav
```

Интеграция стриминга здорова, когда выполняется всё нижеперечисленное:

- Первое сообщение клиента — `ready` с `version: "1.0"`, а выбранная вами
  частота входит в его `supported_rates`.
- `...` partial появляются примерно раз в секунду во время аудио и идут в
  нижнем регистре/сырыми; `>>>` final коммитят обогащённый текст после
  каждой паузы и несут `words[]` с таймингами и (обычно) `confidence`.
- Отправка `stop` даёт ровно ещё один `final` (возможно, пустой), и сокет
  закрывается только после него — остановка на середине слова всё равно
  распознаёт это слово.
- `ready.max_session_secs` / `ready.idle_timeout_secs` совпадают со
  значениями, переданными в `gigastt serve`, и ваша логика
  ротации/backoff (рецепты 4/5) переживает принудительный потолок:
  `gigastt serve --max-session-secs 30`.

## Типичные ошибки

Симптом → причина с указателем на исправление — полная таблица для
оператора живёт в
[docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md),
а таблицы кодов ошибок/закрытия — в
[docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#error-codes);
не отлаживайте по памяти.

| Симптом | Причина | Куда смотреть |
|---|---|---|
| Подключение висит ~30 с, затем `{"code":"timeout","retry_after_ms":30000}` | Насыщение пула — checkout происходит *до* `ready`, поэтому ошибка заменяет его | Рецепт 5; [api.md error codes](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#error-codes) |
| Сокет закрывается 1008 ровно на часовой отметке | Потолок `--max-session-secs`; `final` сбрасывается первым, так что просто переподключайтесь | Рецепт 4; [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| Сокет закрывается 1001 после ~5 мин тишины | Idle-таймаут — ни одного фрейма; шлите тихий PCM, чтобы остаться живыми | Рецепт 4; [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| Сокет закрывается 1009 | Фрейм превысил `--ws-frame-max-bytes` (по умолчанию 512 КиБ) — режьте мельче | Рецепт 1; [api.md limits](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#session-and-frame-limits) |
| Upgrade отклонён с HTTP 503 `{"code":"initializing"}` | Модель ещё скачивается/квантизируется — опрашивайте `/ready`, не перезапускайте | [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| Браузерное приложение с другого origin не подключается | Allowlist origin — по умолчанию только loopback; добавьте `--allow-origin` | [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) |
| Финалы приходят в голом нижнем регистре, без пунктуации | Пунктуационная модель не подключена или политика выключена; `e2e_rnnt` пунктуирует сам | Рецепт 2; [troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md) |
| `configure` не действует | Отправлен после первого аудиофрейма (`configure_too_late`) — шлите сразу после `ready` | Рецепт 1 |
| Транскрипта нет вообще | Три независимых домена отказа: готовность сервера, захват аудио, язык/голова | [триаж troubleshooting](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md#no-transcript-audio-capture-vs-stt-startup-vs-language-config) |

Deepgram-совместимый режим WebSocket (drop-in эндпоинт для клиентов
Deepgram) в работе; эта глава покрывает только нативный протокол.

## Ссылки

- Справочники (канонические, здесь не дублируются):
  [docs/api.md — протокол WebSocket](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#websocket--real-time-streaming),
  [docs/asyncapi.yaml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/asyncapi.yaml),
  [docs/troubleshooting.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/troubleshooting.md),
  [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)
- Клиентский код:
  [examples/](https://github.com/ekhodzitsky/gigastt/tree/main/examples)
  (Python, Bun/TypeScript, Go, Kotlin, Rust),
  [sdks/go](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/go),
  [sdks/js](https://github.com/ekhodzitsky/gigastt/tree/main/sdks/js)
- В этой книге: [Начало работы](01-getting-started.md) — установка и первый
  запуск, [Модели и бэкенды](04-models-and-backends.md) — головы и поведение
  пунктуации/ITN, [Серверная интеграция](05-server-integration.md) —
  альтернативы живому стримингу на REST/SSE/jobs,
  [Развёртывание и эксплуатация](06-deployment-ops.md) — запуск `serve` в
  продакшене.
