import Foundation
import GigasttFFI

/// Errors surfaced by the GigaSTT Swift wrapper.
///
/// The underlying C ABI signals failure with `NULL` returns and does not
/// expose a structured error channel, so these cases describe *where* the
/// failure happened rather than carrying an engine-side message.
public enum GigasttError: Error {
    /// `gigastt_engine_new` / `gigastt_engine_new_with_pool_size` returned `NULL`.
    /// Usually a missing or unreadable model directory.
    case engineLoadFailed(modelDir: String)
    /// `gigastt_stream_new` returned `NULL` (pool checkout failed).
    case streamCreationFailed
    /// A transcription / streaming call returned `NULL`.
    case inferenceFailed
    /// The C function returned a string that was not valid UTF-8, or the
    /// returned JSON could not be decoded into `[TranscriptSegment]`.
    case decodingFailed(underlying: Error?)
}

/// A single recognized word with timing and confidence metadata.
///
/// Field names mirror the engine's `WordInfo` serde representation exactly:
/// `word`, `start`, `end`, `confidence`, and an optional `speaker`.
public struct Word: Codable, Sendable, Equatable {
    /// The recognized word text.
    public let word: String
    /// Start time in seconds from the beginning of the audio stream.
    public let start: Double
    /// End time in seconds from the beginning of the audio stream.
    public let end: Double
    /// Softmax confidence score in `0.0...1.0`.
    public let confidence: Double
    /// Speaker label from diarization (zero-based). `nil` when diarization is off.
    public let speaker: UInt32?

    public init(word: String, start: Double, end: Double, confidence: Double, speaker: UInt32? = nil) {
        self.word = word
        self.start = start
        self.end = end
        self.confidence = confidence
        self.speaker = speaker
    }
}

/// A transcript segment emitted by the streaming engine.
///
/// Mirrors the engine's `TranscriptSegment` serde representation: `text`,
/// `words`, `is_final`, `timestamp`. Partial segments (`isFinal == false`)
/// are interim and may change; final segments are completed utterances.
public struct TranscriptSegment: Codable, Sendable, Equatable {
    /// Recognized text for this segment.
    public let text: String
    /// Individual words with timing and confidence metadata.
    public let words: [Word]
    /// Whether this segment is final (utterance complete) or partial (interim).
    public let isFinal: Bool
    /// Unix timestamp (seconds since epoch) when this segment was produced.
    public let timestamp: Double

    private enum CodingKeys: String, CodingKey {
        case text
        case words
        case isFinal = "is_final"
        case timestamp
    }

    public init(text: String, words: [Word], isFinal: Bool, timestamp: Double) {
        self.text = text
        self.words = words
        self.isFinal = isFinal
        self.timestamp = timestamp
    }
}

/// Decode a JSON array string of `TranscriptSegment` values.
///
/// The engine's `gigastt_stream_process_chunk` / `gigastt_stream_flush`
/// return `serde_json::to_string(&Vec<TranscriptSegment>)`.
private func decodeSegments(_ json: String) throws -> [TranscriptSegment] {
    guard let data = json.data(using: .utf8) else {
        throw GigasttError.decodingFailed(underlying: nil)
    }
    do {
        return try JSONDecoder().decode([TranscriptSegment].self, from: data)
    } catch {
        throw GigasttError.decodingFailed(underlying: error)
    }
}

/// Take ownership of a `char*` returned by the C ABI, copy it into a Swift
/// `String`, and free it via `gigastt_string_free`. Returns `nil` if the
/// pointer is `NULL` (the engine's failure sentinel).
private func takeOwnedCString(_ ptr: UnsafeMutablePointer<CChar>?) -> String? {
    guard let ptr else { return nil }
    defer { gigastt_string_free(ptr) }
    return String(cString: ptr)
}

/// A loaded GigaAM inference engine.
///
/// Wraps the opaque `GigasttEngine` C handle with RAII: `deinit` calls
/// `gigastt_engine_free`. Not thread-safe per the C ABI contract — use one
/// `Engine` (and its `Stream`s) from a single thread at a time, or guard
/// access externally.
public final class Engine {
    /// Non-null for the lifetime of the instance; freed exactly once in `deinit`.
    fileprivate let handle: OpaquePointer

    /// Load the ONNX models from `modelDir` using the default pool size.
    ///
    /// - Throws: ``GigasttError/engineLoadFailed(modelDir:)`` if the engine
    ///   cannot be loaded (missing models, unreadable directory).
    public convenience init(modelDir: String) throws {
        try self.init(modelDir: modelDir, poolSize: nil)
    }

    /// Load the ONNX models from `modelDir` with an explicit session pool size.
    ///
    /// Each pooled session loads the full encoder, so RAM scales with
    /// `poolSize`. On iOS prefer `poolSize: 1` (~350 MB).
    public convenience init(modelDir: String, poolSize: Int) throws {
        try self.init(modelDir: modelDir, poolSize: Optional(poolSize))
    }

    private init(modelDir: String, poolSize: Int?) throws {
        let created: OpaquePointer? = modelDir.withCString { cDir in
            if let poolSize {
                return gigastt_engine_new_with_pool_size(cDir, UInt(poolSize))
            }
            return gigastt_engine_new(cDir)
        }
        guard let created else {
            throw GigasttError.engineLoadFailed(modelDir: modelDir)
        }
        self.handle = created
    }

    /// Transcribe an audio file and return the recognized transcript.
    ///
    /// The C ABI requires `path` to be a relative path inside the current
    /// working directory (absolute paths and `..` are rejected engine-side).
    ///
    /// - Returns: The transcript text (the engine returns plain text here,
    ///   not JSON).
    /// - Throws: ``GigasttError/inferenceFailed`` on a `NULL` return.
    public func transcribeFile(path: String) throws -> String {
        let result: String? = path.withCString { cPath in
            takeOwnedCString(gigastt_transcribe_file(handle, cPath))
        }
        guard let result else {
            throw GigasttError.inferenceFailed
        }
        return result
    }

    deinit {
        gigastt_engine_free(handle)
    }
}

/// A real-time streaming transcription session bound to an ``Engine``.
///
/// Wraps the opaque `GigasttStream` C handle with RAII: `deinit` calls
/// `gigastt_stream_free`, returning the inference triplet to the engine pool.
/// Holds a strong reference to its ``Engine`` so the engine outlives the
/// stream (the C ABI dereferences both handles on every call).
public final class Stream {
    private let engine: Engine
    private let handle: OpaquePointer

    /// Open a new streaming session against `engine`.
    ///
    /// - Throws: ``GigasttError/streamCreationFailed`` if the engine pool
    ///   could not provide an inference session.
    public init(engine: Engine) throws {
        guard let created = gigastt_stream_new(engine.handle) else {
            throw GigasttError.streamCreationFailed
        }
        self.engine = engine
        self.handle = created
    }

    /// Feed a chunk of little-endian mono PCM16 audio at `sampleRate`.
    ///
    /// Audio is resampled to 16 kHz internally when `sampleRate != 16000`.
    ///
    /// - Returns: The transcript segments produced by this chunk (possibly empty).
    /// - Throws: ``GigasttError/inferenceFailed`` on a `NULL` return, or
    ///   ``GigasttError/decodingFailed(underlying:)`` if the JSON is malformed.
    public func processChunk(_ pcm16: Data, sampleRate: UInt32) throws -> [TranscriptSegment] {
        let json: String? = pcm16.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> String? in
            let base = raw.bindMemory(to: UInt8.self).baseAddress
            let ptr = gigastt_stream_process_chunk(engine.handle, handle, base, UInt(pcm16.count), sampleRate)
            return takeOwnedCString(ptr)
        }
        guard let json else {
            throw GigasttError.inferenceFailed
        }
        return try decodeSegments(json)
    }

    /// Signal end-of-stream and drain any final segment(s).
    ///
    /// - Returns: The final transcript segments (possibly empty).
    /// - Throws: ``GigasttError/inferenceFailed`` on a `NULL` return, or
    ///   ``GigasttError/decodingFailed(underlying:)`` if the JSON is malformed.
    public func flush() throws -> [TranscriptSegment] {
        guard let json = takeOwnedCString(gigastt_stream_flush(engine.handle, handle)) else {
            throw GigasttError.inferenceFailed
        }
        return try decodeSegments(json)
    }

    deinit {
        gigastt_stream_free(handle)
    }
}
