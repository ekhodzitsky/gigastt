// WebSocket client for gigastt — streams a WAV file and prints transcription.
//
// Dependencies (Gradle):
//   implementation("com.squareup.okhttp3:okhttp:4.12.0")
//   implementation("org.json:json:20240303")
//
// Compile and run:
//   kotlinc KotlinClient.kt -include-runtime -d client.jar
//   java -jar client.jar <audio.wav> [ws://host:port]

import okhttp3.*
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CountDownLatch

private const val WAV_HEADER_BYTES = 44
private const val CHUNK_BYTES = 32768 // ~1s at 16kHz PCM16

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("Usage: java -jar client.jar <audio.wav> [ws://host:port]")
        System.exit(1)
    }

    val wavPath = args[0]
    val serverBase = if (args.size > 1) args[1] else "ws://127.0.0.1:9876"
    val serverUrl = if (serverBase.endsWith("/ws")) serverBase else "$serverBase/ws"

    val fileBytes = File(wavPath).readBytes()
    if (fileBytes.size <= WAV_HEADER_BYTES) {
        System.err.println("File too small to be a valid WAV")
        System.exit(1)
    }
    val pcm = fileBytes.copyOfRange(WAV_HEADER_BYTES, fileBytes.size)

    val client = OkHttpClient()
    val request = Request.Builder().url(serverUrl).build()
    val latch = CountDownLatch(1)

    val listener = object : WebSocketListener() {
        override fun onOpen(ws: WebSocket, response: Response) {
            // Send audio once connection is open
            pcm.toList().chunked(CHUNK_BYTES).forEach { chunk ->
                ws.send(okio.ByteString.of(*chunk.toByteArray()))
            }
            ws.send("""{"type":"stop"}""")
        }

        override fun onMessage(ws: WebSocket, text: String) {
            val msg = JSONObject(text)
            when (msg.getString("type")) {
                "ready" -> print("Connected: ${msg.optString("model")} @ ${msg.optInt("sample_rate")}Hz\n\n")
                "partial" -> print("\r  ... ${msg.getString("text")}")
                "final" -> {
                    print("\r  >>> ${msg.getString("text")}\n")
                    ws.close(1000, null)
                    latch.countDown()
                }
                "error" -> {
                    System.err.println("\n  ERR: ${msg.optString("message")}")
                    ws.close(1000, null)
                    latch.countDown()
                }
            }
        }

        override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
            System.err.println("WebSocket error: ${t.message}")
            latch.countDown()
        }
    }

    client.newWebSocket(request, listener)
    latch.await()
    client.dispatcher.executorService.shutdown()
}
