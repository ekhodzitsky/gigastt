package com.gigastt

/**
 * Kotlin JNI bridge for the gigastt Russian STT engine.
 *
 * Load `libgigastt.so` from your app's `jniLibs` and call these functions
 * to run on-device speech recognition.
 *
 * Typical lifecycle:
 * ```
 * val engine = GigasttBridge.engineNew(modelDir.absolutePath)
 * if (engine == 0L) { /* handle error */ }
 *
 * val text = GigasttBridge.transcribeFile(engine, wavPath)
 * GigasttBridge.stringFree(text)   // only if you got a non-null string
 *
 * GigasttBridge.engineFree(engine)
 * ```
 */
object GigasttBridge {

    init {
        System.loadLibrary("gigastt")
    }

    /**
     * Load the ONNX models from [modelDir] and create an inference engine.
     *
     * Returns an opaque handle (pointer cast to Long) or 0L on failure.
     */
    @JvmStatic
    external fun engineNew(modelDir: String): Long

    /**
     * Transcribe a WAV file and return the recognized text.
     *
     * The returned string must be freed with [stringFree] when no longer needed.
     * Returns `null` on error.
     */
    @JvmStatic
    external fun transcribeFile(engine: Long, wavPath: String): String?

    /**
     * Create a new real-time streaming session.
     *
     * Returns an opaque stream handle or 0L on failure.
     */
    @JvmStatic
    external fun streamNew(engine: Long): Long

    /**
     * Feed a chunk of PCM16 audio into a streaming session.
     *
     * [pcm16Bytes] must be little-endian mono PCM16 at the given [sampleRate].
     * The audio is resampled to 16 kHz internally if needed.
     *
     * Returns a JSON array of transcript segments (or `null` on error).
     * Each segment has fields: `text`, `is_final`, `timestamp`.
     * The returned string must be freed with [stringFree].
     */
    @JvmStatic
    external fun streamProcessChunk(
        engine: Long,
        stream: Long,
        pcm16Bytes: ByteArray,
        sampleRate: Int
    ): String?

    /**
     * Signal end-of-stream and return the final segment(s).
     *
     * Returns a JSON array (possibly `[]`). The string must be freed with [stringFree].
     */
    @JvmStatic
    external fun streamFlush(engine: Long, stream: Long): String?

    /**
     * Free a streaming session and return its inference triplet to the pool.
     */
    @JvmStatic
    external fun streamFree(stream: Long)

    /**
     * Free a C string previously returned by [transcribeFile], [streamProcessChunk],
     * or [streamFlush].
     */
    @JvmStatic
    external fun stringFree(s: String?)

    /**
     * Tear down the engine and release all ONNX sessions.
     */
    @JvmStatic
    external fun engineFree(engine: Long)
}
