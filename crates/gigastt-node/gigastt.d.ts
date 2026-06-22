/** A recognized word with timing, confidence, and optional speaker label. */
export interface Word {
  text: string
  startS: number
  endS: number
  confidence: number
  speaker?: number
}

/** A transcript segment (interim or final) with its words. */
export interface TranscriptSegment {
  text: string
  words: Array<Word>
  isFinal: boolean
}

/** The full result of transcribing a file. */
export interface Transcript {
  text: string
  words: Array<Word>
  durationS: number
}

/**
 * On-device speech-recognition engine. Loads the GigaAM v3 model from a
 * side-loaded directory. Thread-safe; share one instance across streams.
 */
export declare class Engine {
  /** Load the model from `modelDir` with an optional session-pool size. */
  constructor(modelDir: string, poolSize?: number | undefined | null)
  /** Transcribe an audio file (WAV / MP3 / M4A / OGG / FLAC) to text + timings. */
  transcribeFile(path: string): Promise<Transcript>
}

/** A streaming transcription session bound to an {@link Engine}. */
export declare class Stream {
  constructor(engine: Engine)
  /** Feed little-endian mono PCM16; `sampleRate` is resampled to 16 kHz. */
  processChunk(pcm16: Uint8Array, sampleRate: number): Promise<Array<TranscriptSegment>>
  /** Flush buffered audio and return any final segment(s). */
  flush(): Promise<Array<TranscriptSegment>>
}
