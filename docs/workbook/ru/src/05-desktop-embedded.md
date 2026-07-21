# Десктоп и встраивание: Swift/SPM, sidecar, Electron, UniFFI

## Сценарий

Вы пишете десктопное или мобильное приложение с локальным распознаванием
русской речи: диктофон на Swift под macOS, записывающее приложение для встреч
на Electron или инструмент на Kotlin / Python. Модель должна работать локально
(без облака), и движок можно поставить двумя способами — встроить **в процесс**
через нативный биндинг или запустить `gigastt serve` как **sidecar**-подпроцесс
и общаться с ним по loopback HTTP/WS API. Эта глава помогает выбрать путь и
проводит по каждому до первого транскрипта.

## Какой путь выбрать: embedded или sidecar

| | Embedded (в процессе) | Sidecar (`gigastt serve` + клиент) |
|---|---|---|
| Как | Движок слинкован с приложением: SwiftPM `GigaSTT`, npm `gigastt`, PyPI `gigastt`, UniFFI-биндинги | Движок работает дочерним процессом; приложение общается по WS/REST через loopback-порт |
| Интерфейс | Нативные вызовы; типизированные ошибки (`throws` / исключения / rejected promises) | Сетевой протокол (`/v1/ws`, REST `/v1/transcribe`, SSE) |
| Деплой | Одна установка пакета; без супервизии процессов | Поиск бинаря на машине пользователя, spawn, супервизия, выбор порта |
| Память | Модель и пул живут внутри вашего приложения (~350–400 МБ RSS на сессию пула) | Модель живёт в отдельном серверном процессе |
| Конкурентность | Один движок/пул на процесс приложения | Один сервер на несколько приложений/клиентов |
| Изоляция падений | Падение движка роняет приложение (и наоборот) | Падение сервера изолировано; приложение выживает и может его перезапустить |
| Обновления | Передеплой приложения с новым движком | Бинарь сервера обновляется независимо от клиентов |
| Версионирование | Приложение и движок — один артефакт | Приложение должно гейтоваться на версию найденного сервера (`/health` → `version`) |

Правила выбора:

- **Embedded**, когда приложение владеет всем аудио-конвейером (диктофон,
  рекордер) и является единственным потребителем. На iOS это *единственный*
  вариант — приложения не могут запускать подпроцессы.
- **Sidecar**, когда один движок делят несколько клиентов, когда приложение
  должно переживать падение движка или когда движок нужно обновлять без
  передеплоя приложения. Оба подтверждённых внешних десктоп-интегратора
  используют именно этот паттерн.

## Предпосылки

- **Директория с моделью** — все рецепты ниже её предполагают. Один раз
  скачайте преквантизованный INT8-бандл (~215 МБ, без скачивания FP32 и без
  квантизации на устройстве):

  ```sh
  gigastt download --prequantized          # -> ~/.gigastt/models
  ```

- **Embedded**: тулчейн вашего биндинга — Xcode 15+ (Swift), Node.js
  (npm-пакет), Python 3 (wheel) или Android SDK/NDK (Kotlin).
- **Sidecar**: бинарь `gigastt` — забандленный в ресурсы приложения или
  установленный (`brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt && brew install gigastt`,
  тарбол релиза или `cargo install gigastt`) — плюс `curl` для проб.

## Рецепт — Swift/SPM (iOS + macOS)

Swift-пакет `GigaSTT` оборачивает C ABI в безопасный Swift-интерфейс. Нативный
код поставляется готовым `GigasttFFI.xcframework` (iOS device `arm64`,
симулятор `arm64`/`x86_64`, macOS `arm64`) со статически слинкованным ONNX
Runtime — отдельный рантайм бандлить не нужно. Требования: iOS 15 / macOS 13
(только Apple Silicon — слайса для Intel macOS нет) и Xcode 15+.

1. **Добавьте пакет.** Xcode → File → Add Package Dependencies… → введите URL
   зеркала `https://github.com/ekhodzitsky/gigastt-swift` и добавьте продукт
   `GigaSTT` к вашему таргету. Зеркало — каноничный удалённый источник (SwiftPM
   требует `Package.swift` в корне репозитория, поэтому поддиректорию монорепо
   `packaging/swift` нельзя подключить по URL — используйте её только как
   локальную path-зависимость для разработки).
2. **Забандлите модель.** Скопируйте `~/.gigastt/models` в таргет приложения
   как **folder reference** (синяя папка — сохраняет структуру директорий;
   жёлтая «group» плющит файлы, и движок их не найдёт). Альтернатива —
   скачать модель при первом запуске и закешировать.
3. **Загрузите движок и транскрибируйте:**

   ```swift
   import GigaSTT

   guard let modelDir = Bundle.main.url(
       forResource: "models", withExtension: nil
   )?.path else {
       fatalError("bundle the model directory as a folder reference")
   }

   // poolSize: 1 keeps RAM around ~350 MB, recommended on device.
   let engine = try Engine(modelDir: modelDir, poolSize: 1)

   // Path is relative to the current working directory; absolute paths and
   // ".." are rejected by the engine.
   let text = try engine.transcribeFile(path: "audio.wav")
   print(text)
   ```

4. **Стриминг** — чанки little-endian mono PCM16 на частоте захвата (внутри
   ресемплируется в 16 кГц):

   ```swift
   let stream = try Stream(engine: engine)

   // pcm16: Data of little-endian Int16 mono samples at 48 kHz.
   for segment in try stream.processChunk(pcm16, sampleRate: 48000) {
       print(segment.text, segment.isFinal)
   }

   // Drain the tail at end-of-stream.
   for segment in try stream.flush() {
       print(segment.text)
   }
   ```

   `processChunk` и `flush` возвращают `[TranscriptSegment]`; каждый сегмент
   несёт `text`, `words` (по каждому слову `word`/`start`/`end`/`confidence` и
   опциональный `speaker`), `isFinal` и `timestamp`.

5. **Обрабатывайте ошибки.** Обёртка бросает `GigasttError`:
   `engineLoadFailed(modelDir:)` (директория модели отсутствует/нечитаема),
   `streamCreationFailed` (не удалось занять сессию пула), `inferenceFailed` и
   `decodingFailed(underlying:)`. C ABI сигналит об ошибке `NULL`-возвратом,
   поэтому кейс говорит, *где* произошёл сбой, а не несёт сообщение движка.

Проверка: запустите приложение, транскрибируйте заведомо известный WAV и
убедитесь, что напечатан ожидаемый текст. Если движок бросает
`engineLoadFailed` при старте — директории модели нет там, куда указывает
`Bundle.main.url(forResource: "models", withExtension: nil)`; см. «Частые
ошибки».

## Рецепт — sidecar-сервер (macOS / Electron)

Запускаем `gigastt serve` как управляемый дочерний процесс. Полный жизненный
цикл: найти бинарь, предустановить модель, запустить, дождаться готовности,
транскрибировать, корректно остановить.

1. **Найдите бинарь** в порядке приоритета: env-override (например,
   `MYAPP_GIGASTT_BIN`) → копия в ресурсах приложения →
   `/opt/homebrew/bin/gigastt` → `/usr/local/bin/gigastt` → `PATH`.
   Залогируйте, какой вариант выбран.
2. **Предустановите модель** при установке или первом запуске, с
   машиночитаемым прогрессом для UI:

   ```sh
   gigastt download --prequantized --progress json
   ```

   stdout несёт по одному NDJSON-событию на строку
   (`{"phase":"download","file":...,"bytes_done":N,"bytes_total":M}`, затем
   `verify`, затем `done`), а коды выхода различают сетевые/дисковые/контрольные
   ошибки — см. [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md).
   `--prequantized` пропускает ~2-минутный проход INT8-квантизации на
   устройстве, поэтому первый `serve` стартует за секунды, а не минуты.
3. **Выберите порт.** Сегодня: фиксированный высокий порт (например, `49876`).
   Автовыбор эфемерного порта (`--port 0` с машиночитаемой строкой
   `LISTENING`), а также `--die-with-parent` и `--log-file` — планируемые
   дополнения сервера: они требуют версии gigastt, где эти флаги есть, поэтому
   пока не полагайтесь на их синтаксис; используйте фиксированный порт и
   жизненный цикл ниже.
4. **Запустите и собирайте логи.** Никогда не отправляйте вывод sidecar'я в
   `/dev/null` — перенаправьте stdout/stderr в лог-файл (если подключаете
   пайпы вместо файла, непрерывно их читайте: полный буфер пайпа может
   заблокировать дочерний процесс). Пример для главного процесса Electron /
   Node:

   ```js
   import { spawn } from 'node:child_process';
   import fs from 'node:fs';

   const PORT = 49876;
   const BASE = `http://127.0.0.1:${PORT}`;

   const log = fs.createWriteStream('gigastt-sidecar.log', { flags: 'a' });
   const server = spawn(gigasttBin, [
     'serve',
     '--port', String(PORT),
     '--pool-size', '1',
     '--model-dir', modelDir,
   ], { stdio: ['ignore', log, log] });

   async function waitForReady(timeoutMs = 120_000) {
     const deadline = Date.now() + timeoutMs;
     for (;;) {
       try {
         const res = await fetch(`${BASE}/ready`);
         // 200 {"status":"ready","pool_available":N,"pool_total":M}
         if (res.ok) return;
         // 503 {"status":"not_ready","reason":"initializing"} — keep waiting.
       } catch {
         // Connection refused: the listener is not up (yet) — or the process
         // is gone. Distinguish by the child exit code, not by killing it.
         if (server.exitCode !== null) {
           throw new Error(`gigastt exited with code ${server.exitCode}`);
         }
       }
       if (Date.now() > deadline) throw new Error('readiness timeout');
       await new Promise((resolve) => setTimeout(resolve, 500));
     }
   }
   ```

5. **Гейт по `/ready`, а не по TCP-connect.** Сервер биндит порт сразу и
   отвечает на пробы из bootstrap-ответчика, пока модель грузится, поэтому
   «порт слушается» не значит «готов». Поллите `GET /ready` до 200; тело 503
   несёт
   `{"status":"not_ready","reason":"initializing"|"pool_exhausted"|"shutting_down"}`.
   Connection-refused означает, что процесс мёртв или ещё не запущен, — а не
   что он завис.
6. **Версионное рукопожатие по HTTP.** `GET /health` возвращает 200 в *обеих*
   фазах — `{"status":"ok","model":"loading","version":"2.13.0"}` во время
   bootstrap, затем `{"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"2.13.0","punctuation":true,"itn":true}`.
   Гейт минимальной версии движка делайте по полю `version`, а не запуском
   `gigastt --version` подпроцессом.
7. **Транскрибация.** Для live-частичных результатов откройте
   WebSocket-сессию на `/v1/ws` (паттерны — в главе
   [Стриминг по WebSocket](04-streaming-ws.md)); для целых файлов — POST на `/v1/transcribe`
   (рецепты — в главе [CLI и пакетная обработка](02-cli-batch.md)). При
   насыщении пула сервер отвечает 503 + `Retry-After` (REST) или ошибкой с
   `retry_after_ms` (WS) — соблюдайте подсказку вместо выдумывания своего
   backoff'а. Перед закрытием WS-сессии отправьте `{"type":"stop"}`, чтобы
   сервер дописал хвост в финальный сегмент.
8. **Корректная остановка.** Пошлите SIGTERM: сервер дожидается завершения
   активных сессий до `--shutdown-drain-secs` (по умолчанию 10), шлёт `final` и
   закрывает WS-клиентов кодом 1001. Эскалируйте до SIGKILL только после окна
   drain. Отслеживайте pid дочернего процесса и завершайте его при выходе
   приложения — осиротевший sidecar держит ~1 ГБ RSS и порт.

Проверка:

```sh
gigastt serve --port 49876 --pool-size 1 &
curl -s http://127.0.0.1:49876/ready    # -> {"status":"ready","pool_available":1,"pool_total":1}
curl -s http://127.0.0.1:49876/health   # -> {"status":"ok","model":"gigaam-v3-rnnt",...,"version":"..."}
kill %1                                  # SIGTERM — process exits within the drain window
```

## Рецепт — Node/Electron в процессе

npm-пакет `gigastt` (napi-rs) встраивает движок внутрь вашего Node/Electron
процесса — без sidecar'я, порта и версионного гейта. Инференс идёт на рабочем
потоке libuv, поэтому вызовы возвращают Promise и не блокируют event loop;
onnxruntime слинкован статически, поэтому `.node`-аддон самодостаточен.

1. **Установка:** `npm install gigastt`. postinstall скачивает ровно один
   готовый бинарь (`gigastt.<platform>.node`, ~47 МБ) под платформу установки
   из GitHub-релиза. Готовые платформы: `darwin-arm64`, `linux-x64-gnu`,
   `linux-arm64-gnu`, `win32-x64-msvc` (Intel macOS нет).
2. **Загрузите модель** — она не забандлена: `gigastt download --prequantized`
   → `~/.gigastt/models`, либо задайте `GIGASTT_MODEL_DIR`.
3. **Использование:**

   ```js
   const { Engine, Stream } = require('gigastt');

   const engine = new Engine('/path/to/gigastt/models'); // new Engine(modelDir, poolSize?)
   const t = await engine.transcribeFile('recording.wav');
   console.log(t.text, t.durationS);
   for (const w of t.words) console.log(w.text, w.startS, w.endS, w.confidence);

   // streaming — await each chunk before sending the next to keep order
   const s = new Stream(engine);
   for (const seg of await s.processChunk(pcm16, 16000)) console.log(seg.text);
   console.log((await s.flush()).map((seg) => seg.text));

   // errors are thrown JS Errors whose message starts with a stable code
   try { new Engine('/no/such/dir'); } catch (e) { /* e.message starts "ModelNotFound:" */ }
   ```

4. **Схема для Electron:** создавайте `Engine` в **главном процессе** (никогда
   в renderer'е) и держите по одному `Stream` на аудиоканал (например, mic +
   system). Каждый `Stream` держит одну сессию пула всё своё время жизни,
   поэтому `poolSize` должен покрывать число живых стримов — третий `Stream`
   на пуле из 2 бросит `PoolExhausted`. Полный двухканальный паттерн с
   IPC-обработчиками — в
   [examples/electron_main.mjs](https://github.com/ekhodzitsky/gigastt/blob/main/examples/electron_main.mjs).
5. **Размер thread-пула.** Инференс идёт на рабочем пуле libuv (по умолчанию 4
   потока, общие с `fs`/`crypto`). Для `N` одновременно транскрибирующих
   каналов запускайте с `UV_THREADPOOL_SIZE >= N`
   (например, `UV_THREADPOOL_SIZE=4 electron .`) и соответствующим `poolSize`.
6. **Упаковка (asar):** держите директорию модели *и* нативный `.node`-аддон
   **вне asar-архива** — нативный код мапит веса в память и делает `dlopen`
   аддона, а оба этих действия невозможны из виртуальной ФС asar. В
   electron-builder используйте `asarUnpack` для `**/*.node` и поставляйте
   модель через `extraResources`.

Проверка: в чекауте репозитория `GIGASTT_MODEL_DIR=~/.gigastt/models npm run smoke`
(из `crates/gigastt-node`) транскрибирует тестовую фикстуру; в вашем
приложении — дождитесь `engine.transcribeFile` на заведомо известном WAV и
проверьте текст.

## Рецепт — Kotlin и Python (UniFFI)

`crates/gigastt-uniffi` генерирует идиоматичные биндинги для Swift, Kotlin и
Python из одного Rust-источника через UniFFI. Они оборачивают синхронное
урезанное ядро (модели подгружаются с диска, без tokio-рантайма), дают
типизированные ошибки `GigasttFfiError` (`ModelNotFound`, `InvalidAudio`,
`PoolExhausted`, `Inference`, `InvalidArgument`) и управляют объектами по
счётчику ссылок.

Общая поверхность (регистр имён следует идиомам языка):

- `Engine(model_dir)` / `Engine.new_with_pool_size(model_dir, pool_size)` ·
  `transcribe_file(path) -> Transcript { text, words, duration_s }`
- `Stream(engine)` · `process_chunk(pcm16, sample_rate) -> [TranscriptSegment]` ·
  `flush() -> [TranscriptSegment]`
- `TranscriptSegment { text, words, is_final }` ·
  `Word { text, start_s, end_s, confidence, speaker }`

**Python — опубликован.** `pip install gigastt` ставит самодостаточный готовый
wheel (`py3-none-<platform>`; onnxruntime слинкован статически):

```python
import gigastt_uniffi as g

engine = g.Engine("/path/to/gigastt/models")        # side-loaded model dir
t = engine.transcribe_file("recording.wav")
print(t.text)
for w in t.words:
    print(w.text, w.start_s, w.end_s, w.confidence)

# streaming
s = g.Stream(engine)
for seg in s.process_chunk(pcm16_bytes, 16000):
    print(seg.text)
print([seg.text for seg in s.flush()])

# typed errors
try:
    g.Engine("/no/such/dir")
except g.GigasttFfiError.ModelNotFound as e:
    ...
```

**Kotlin — экспериментальный AAR.** Кросс-сборка Rust доказана (CI
кросс-компилирует `libgigastt_uniffi.so` под каждый ABI через cargo-ndk), но
сборка и публикация Gradle/Maven AAR ещё не проверены end-to-end. Локальная
сборка:

```sh
# native libs per ABI
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 \
  -o packaging/android/gigastt/src/main/jniLibs build --release -p gigastt-uniffi
# Kotlin bindings (from a host build of the cdylib; metadata is arch-independent)
cargo build --release -p gigastt-uniffi
cargo run --release -p gigastt-uniffi --bin uniffi-bindgen -- generate \
  --library target/release/libgigastt_uniffi.* --language kotlin \
  --out-dir packaging/android/gigastt/src/main/kotlin
# assemble
cd packaging/android && gradle :gigastt:assembleRelease
```

**Swift (UniFFI):** биндинги генерируются (`--language swift`), но поставляемый
SwiftPM-пакет использует рукописную обёртку над C ABI из Swift-рецепта выше —
для разработки приложений берите её.

Перегенерация любого биндинга из чекаута:

```sh
cargo build -p gigastt-uniffi
LIB=target/debug/libgigastt_uniffi.dylib   # .so on Linux

cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language python --out-dir bindings/python
cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language swift  --out-dir bindings/swift
cargo run -p gigastt-uniffi --bin uniffi-bindgen -- generate --library "$LIB" --language kotlin --out-dir bindings/kotlin
```

Сгенерированные биндинги — артефакты сборки (`bindings/` в git-игноре). Статус
и детали упаковки:
[crates/gigastt-uniffi/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-uniffi/README.md),
[packaging/android/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/android/README.md).

Проверка: Python-сниппет выше печатает транскрипт заведомо известного файла;
для Kotlin — соберите AAR и транскрибируйте из instrumented-теста или
отладочного экрана.

## Проверка результата

Сквозной чек-лист для любого из путей:

- **Первый транскрипт**: приложение печатает ожидаемый текст для известного
  WAV (любой файл с русской речью, например сэмпл Golos).
- **Модель на диске**: `ls ~/.gigastt/models` показывает файлы `v3_rnnt_*`
  (бандл `--prequantized` включает `v3_rnnt_encoder_int8.onnx`).
- **Память**: RSS в ожидаемых пределах — примерно 350–400 МБ на сессию пула;
  `poolSize`/`pool-size` 1 — правильный дефолт для устройства.
- **Здоровье sidecar'я**: `curl -s http://127.0.0.1:<port>/ready` возвращает
  `{"status":"ready",...}`, а `/health` показывает `version`, на которую вы
  гейтованы; SIGTERM завершает процесс в пределах окна drain с чистым хвостом
  лога.
- **Порядок в стриминге**: частичные сегменты (`isFinal == false`) могут
  пересматриваться; сохраняйте только финальные, а перед завершением сессии
  вызывайте `flush()` (или шлите `{"type":"stop"}` по WS), чтобы не потерять
  хвост.

## Частые ошибки

- **Модель забандлена не как folder reference (Swift).** Добавленная жёлтой
  «group», Xcode плющит директорию, и движок падает при старте с
  `engineLoadFailed`. Исправление: добавьте `models` как folder reference
  (синяя папка) и в отладочном запуске проверьте, что
  `Bundle.main.url(forResource: "models", withExtension: nil)` не nil.
- **Неопределённые символы `std::__1::*` при линковке (Swift).** ONNX Runtime —
  это C++; актуальный пакет уже объявляет `.linkedLibrary("c++")` — такая
  ошибка означает, что вы используете старый или самосборный xcframework.
  Берите релизный пакет из зеркала (или актуальный `packaging/swift`).
- **Блокирующий вызов в UI-потоке.** Вызовы `Engine`/`Stream` синхронны, а
  хендлы не thread-safe. Выполняйте всю работу движка на выделенной фоновой
  очереди/акторе (Swift) и сериализуйте доступ; никогда не транскрибируйте на
  главном потоке. В Node вызовы уже идут на пуле libuv — просто делайте
  `await`.
- **Таймауты готовности при первом запуске (sidecar).** Обычный
  `gigastt download` оставляет ~2-минутный проход INT8-квантизации на первый
  `serve`, а модель пунктуации подгружается лениво при первом старте — клиент
  с таймаутом загрузки 10–30 с сдаётся слишком рано. Исправление:
  предустановка через `gigastt download --prequantized`, щедрый таймаут
  готовности и ветвление по `reason` из `/ready` вместо убийства процесса.
- **Убийство сервера на 503.** Пока модель грузится, порт занят
  bootstrap-ответчиком и каждый API отвечает 503 `initializing` — это
  прогресс, а не зависание. Признак сбоя — только connection-refused вместе с
  мёртвым дочерним процессом.
- **Коллизия захардкоженного порта.** Занятый фиксированный порт (часто —
  осиротевшим gigastt от прошлого запуска) проявляется как молчаливый таймаут.
  Проверяйте `/health` → `version`, чтобы понять, ваш ли это сервер, и
  завершайте дочерний процесс при выходе приложения (флаг `--die-with-parent`
  в планах).
- **Логи sidecar'я в `/dev/null`.** Они понадобятся в первый же раз, когда
  директория модели окажется битой. Перенаправляйте stdout/stderr в файл — а
  если используете пайпы, непрерывно их читайте во избежание дедлока на буфере
  пайпа.
- **`require('gigastt')` падает после `npm install --ignore-scripts`.**
  postinstall, скачивающий нативный бинарь, был пропущен; запустите `node
  install.js` внутри пакета.
- **Модель или аддон внутри asar (Electron).** Нативный код мапит веса в
  память и делает `dlopen` для `.node` — обоим нужны реальные файлы.
  `asarUnpack` для аддона и поставка модели через `extraResources`.
- **Абсолютный путь в `transcribeFile` (Swift).** C ABI принимает только
  относительные пути внутри рабочей директории — сначала направьте рабочую
  директорию на расположение файла (или скопируйте файл туда).

## Ссылки

В этой книге:

- [Начало работы](01-getting-started.md) — установка и скачивание модели
- [Стриминг по WebSocket](04-streaming-ws.md) — паттерны WebSocket-протокола для sidecar'я
- [CLI и пакетная обработка](02-cli-batch.md) — рецепты REST / SSE / jobs

Справочные материалы (не дублируем — читайте здесь):

- [packaging/swift/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/swift/README.md) — справочник API Swift-пакета
- [crates/gigastt-node/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-node/README.md) — API npm-пакета + trade-offs in-process vs sidecar
- [crates/gigastt-uniffi/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/crates/gigastt-uniffi/README.md) — UniFFI-биндинги и генерация
- [packaging/android/README.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/android/README.md) — статус сборки Kotlin/AAR
- [examples/electron_main.mjs](https://github.com/ekhodzitsky/gigastt/blob/main/examples/electron_main.mjs) — полный паттерн главного процесса Electron
- [docs/quickstarts.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/quickstarts.md) — квикстарты по биндингам и таблица доступности
- [docs/embedding-packaging.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/embedding-packaging.md) — статическая vs динамическая линковка onnxruntime
- [docs/api.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md) — справочник `/health`, `/ready`, REST/WS/SSE
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — флаги `serve` и `download` (использованные выше ручки жизненного цикла)

Отдельный гайд по встраиванию и дистрибуции (`docs/embedding.md`, включая
темы вроде нотаризации macOS) находится в работе.
