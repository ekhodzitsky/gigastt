# Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add speaker identification to gigastt — label each word/segment with a speaker ID using a separate speaker embedding ONNX model + online clustering. Works in both WebSocket streaming and HTTP transcription. Optional feature behind `--features diarization` compile flag.

**Architecture:** WeSpeaker ResNet34 ONNX model (26.5MB) extracts 256-dim speaker embeddings from ~1.5s audio segments. Online incremental clustering (cosine similarity threshold) assigns speaker IDs in real-time. Model downloaded separately via `gigastt download --diarization`. All diarization code gated by `#[cfg(feature = "diarization")]`.

**Tech Stack:** ort (ONNX Runtime, already in deps), wespeaker-voxceleb-resnet34 ONNX model from HuggingFace

---

### Task 1: Add `diarization` cargo feature

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add feature flag**

In `Cargo.toml` `[features]` section, add:

```toml
diarization = []
```

This is a pure Rust feature (no extra crate deps — reuses existing `ort`).

- [ ] **Step 2: Verify compilation with and without feature**

Run: `cargo check --features coreml` (without diarization)
Run: `cargo check --features coreml,diarization`
Expected: both compile.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "chore: add diarization cargo feature flag"
```

---

### Task 2: Speaker model download

**Files:**
- Modify: `src/model/mod.rs` (add speaker model download)
- Modify: `src/main.rs` (add `--diarization` flag to download command)

- [ ] **Step 1: Add speaker model constants**

In `src/model/mod.rs`, add constants for the speaker embedding model:

```rust
#[cfg(feature = "diarization")]
const SPEAKER_MODEL_REPO: &str = "onnx-community/wespeaker-voxceleb-resnet34-LM";
#[cfg(feature = "diarization")]
const SPEAKER_MODEL_FILE: &str = "wespeaker_resnet34.onnx";
```

- [ ] **Step 2: Add download function for speaker model**

```rust
#[cfg(feature = "diarization")]
pub async fn ensure_speaker_model(model_dir: &str) -> anyhow::Result<()> {
    let dir = Path::new(model_dir);
    let model_path = dir.join(SPEAKER_MODEL_FILE);
    if model_path.exists() {
        tracing::info!("Speaker model already present at {}", model_path.display());
        return Ok(());
    }
    tracing::info!("Downloading speaker embedding model...");
    // Download from HuggingFace using same pattern as STT model
    download_hf_file(SPEAKER_MODEL_REPO, "onnx/model.onnx", &model_path).await?;
    tracing::info!("Speaker model saved to {}", model_path.display());
    Ok(())
}
```

This reuses the existing HuggingFace download infrastructure in model/mod.rs.

- [ ] **Step 3: Add `--diarization` flag to CLI download command**

In `src/main.rs`, extend the `Download` command:

```rust
Download {
    #[arg(long, default_value_t = model::default_model_dir())]
    model_dir: String,
    /// Also download the speaker diarization model
    #[cfg(feature = "diarization")]
    #[arg(long)]
    diarization: bool,
},
```

And in the match arm:

```rust
Commands::Download { model_dir, #[cfg(feature = "diarization")] diarization } => {
    model::ensure_model(&model_dir).await?;
    #[cfg(feature = "diarization")]
    if diarization {
        model::ensure_speaker_model(&model_dir).await?;
    }
    tracing::info!("Model ready at {model_dir}");
}
```

- [ ] **Step 4: Test compilation**

Run: `cargo check --features coreml,diarization`
Expected: compiles.

- [ ] **Step 5: Commit**

```bash
git add src/model/mod.rs src/main.rs
git commit -m "feat(model): add speaker embedding model download (--diarization flag)"
```

---

### Task 3: Speaker embedding extraction module

**Files:**
- Create: `src/inference/diarization.rs`
- Modify: `src/inference/mod.rs` (add conditional module)

- [ ] **Step 1: Create diarization module with SpeakerEncoder**

Create `src/inference/diarization.rs`:

```rust
//! Speaker diarization: embedding extraction and online clustering.
//!
//! Uses WeSpeaker ResNet34 ONNX model (26.5MB) for 256-dim speaker embeddings.
//! Online incremental clustering assigns speaker IDs in real-time.

use anyhow::Context;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::ort_err;

const EMBEDDING_DIM: usize = 256;
const SEGMENT_SAMPLES: usize = 24000; // 1.5s at 16kHz
const COSINE_THRESHOLD: f32 = 0.5; // similarity threshold for same speaker

/// Speaker embedding extractor using WeSpeaker ResNet34 ONNX model.
pub struct SpeakerEncoder {
    session: Session,
}

impl SpeakerEncoder {
    /// Load the speaker embedding ONNX model.
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let model_path = model_dir.join("wespeaker_resnet34.onnx");
        if !model_path.exists() {
            anyhow::bail!(
                "Speaker model not found at {}. Run `gigastt download --diarization` first.",
                model_path.display()
            );
        }

        // Use same EP configuration as STT models
        let session = Session::builder()
            .map_err(ort_err)?
            .commit_from_file(&model_path)
            .map_err(ort_err)?;

        tracing::info!("Speaker embedding model loaded");
        Ok(Self { session })
    }

    /// Extract 256-dim speaker embedding from audio segment (16kHz f32 samples).
    ///
    /// Input should be ~1.5s of audio (24000 samples). Shorter segments are zero-padded.
    pub fn extract_embedding(&self, samples: &[f32]) -> anyhow::Result<[f32; EMBEDDING_DIM]> {
        // Pad or truncate to SEGMENT_SAMPLES
        let mut input = vec![0.0f32; SEGMENT_SAMPLES];
        let len = samples.len().min(SEGMENT_SAMPLES);
        input[..len].copy_from_slice(&samples[..len]);

        let input_shape = [1, SEGMENT_SAMPLES as i64];
        let input_tensor = TensorRef::from_array_view(
            (input_shape.as_slice(), input.as_slice())
        ).map_err(ort_err)?;

        let outputs = self.session
            .run(ort::inputs![input_tensor].map_err(ort_err)?)
            .map_err(ort_err)?;

        let embedding_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;

        let embedding_view = embedding_tensor.view();
        let mut embedding = [0.0f32; EMBEDDING_DIM];
        let slice = embedding_view.as_slice().context("Non-contiguous embedding tensor")?;
        let len = slice.len().min(EMBEDDING_DIM);
        embedding[..len].copy_from_slice(&slice[..len]);

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        Ok(embedding)
    }
}

/// Online speaker clustering using cosine similarity.
pub struct SpeakerCluster {
    /// Centroids: average embedding per speaker.
    centroids: Vec<[f32; EMBEDDING_DIM]>,
    /// Number of embeddings merged into each centroid.
    counts: Vec<usize>,
}

impl SpeakerCluster {
    pub fn new() -> Self {
        Self {
            centroids: Vec::new(),
            counts: Vec::new(),
        }
    }

    /// Assign a speaker ID to an embedding. Creates new speaker if no match.
    pub fn assign(&mut self, embedding: &[f32; EMBEDDING_DIM]) -> u32 {
        // Find closest existing centroid
        let mut best_id = None;
        let mut best_sim = COSINE_THRESHOLD;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, centroid);
            if sim > best_sim {
                best_sim = sim;
                best_id = Some(i);
            }
        }

        match best_id {
            Some(id) => {
                // Update centroid (running average)
                let count = self.counts[id] as f32;
                let new_count = count + 1.0;
                for j in 0..EMBEDDING_DIM {
                    self.centroids[id][j] =
                        (self.centroids[id][j] * count + embedding[j]) / new_count;
                }
                self.counts[id] += 1;
                id as u32
            }
            None => {
                // New speaker
                let id = self.centroids.len() as u32;
                self.centroids.push(*embedding);
                self.counts.push(1);
                id
            }
        }
    }

    /// Number of speakers identified so far.
    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }
}

fn cosine_similarity(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-6 || norm_b < 1e-6 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let mut a = [0.0f32; EMBEDDING_DIM];
        a[0] = 1.0;
        a[1] = 0.5;
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = [0.0f32; EMBEDDING_DIM];
        let mut b = [0.0f32; EMBEDDING_DIM];
        a[0] = 1.0;
        b[1] = 1.0;
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let mut a = [0.0f32; EMBEDDING_DIM];
        let mut b = [0.0f32; EMBEDDING_DIM];
        a[0] = 1.0;
        b[0] = -1.0;
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cluster_new_speaker() {
        let mut cluster = SpeakerCluster::new();
        let mut emb = [0.0f32; EMBEDDING_DIM];
        emb[0] = 1.0;
        let id = cluster.assign(&emb);
        assert_eq!(id, 0);
        assert_eq!(cluster.num_speakers(), 1);
    }

    #[test]
    fn test_cluster_same_speaker() {
        let mut cluster = SpeakerCluster::new();
        let mut emb1 = [0.0f32; EMBEDDING_DIM];
        emb1[0] = 1.0;
        emb1[1] = 0.1;
        let id1 = cluster.assign(&emb1);

        // Very similar embedding → same speaker
        let mut emb2 = [0.0f32; EMBEDDING_DIM];
        emb2[0] = 0.99;
        emb2[1] = 0.12;
        let id2 = cluster.assign(&emb2);

        assert_eq!(id1, id2);
        assert_eq!(cluster.num_speakers(), 1);
    }

    #[test]
    fn test_cluster_different_speakers() {
        let mut cluster = SpeakerCluster::new();

        // Speaker 0: dominant in dimension 0
        let mut emb1 = [0.0f32; EMBEDDING_DIM];
        emb1[0] = 1.0;
        let id1 = cluster.assign(&emb1);

        // Speaker 1: dominant in dimension 100 (orthogonal)
        let mut emb2 = [0.0f32; EMBEDDING_DIM];
        emb2[100] = 1.0;
        let id2 = cluster.assign(&emb2);

        assert_ne!(id1, id2);
        assert_eq!(cluster.num_speakers(), 2);
    }

    #[test]
    fn test_cluster_three_speakers() {
        let mut cluster = SpeakerCluster::new();

        let mut emb_a = [0.0f32; EMBEDDING_DIM];
        emb_a[0] = 1.0;
        let id_a = cluster.assign(&emb_a);

        let mut emb_b = [0.0f32; EMBEDDING_DIM];
        emb_b[50] = 1.0;
        let id_b = cluster.assign(&emb_b);

        let mut emb_c = [0.0f32; EMBEDDING_DIM];
        emb_c[150] = 1.0;
        let id_c = cluster.assign(&emb_c);

        // Back to speaker A
        let mut emb_a2 = [0.0f32; EMBEDDING_DIM];
        emb_a2[0] = 0.98;
        emb_a2[1] = 0.05;
        let id_a2 = cluster.assign(&emb_a2);

        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);
        assert_eq!(id_c, 2);
        assert_eq!(id_a2, id_a); // recognized as speaker A
        assert_eq!(cluster.num_speakers(), 3);
    }
}
```

- [ ] **Step 2: Add module declaration in inference/mod.rs**

In `src/inference/mod.rs`, add after `pub mod audio;`:

```rust
#[cfg(feature = "diarization")]
pub mod diarization;
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib diarization --features coreml,diarization`
Expected: all 7 clustering/similarity tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/inference/diarization.rs src/inference/mod.rs
git commit -m "feat(diarization): speaker embedding extractor and online clustering"
```

---

### Task 4: Extend protocol with speaker field

**Files:**
- Modify: `src/protocol/mod.rs` (Ready diarization flag, Configure diarization option)
- Modify: `src/inference/mod.rs` (WordInfo speaker field)

- [ ] **Step 1: Add `speaker` field to WordInfo**

In `src/inference/mod.rs`, find the `WordInfo` struct and add:

```rust
/// Speaker ID (0-indexed). Present only when diarization is enabled.
#[serde(skip_serializing_if = "Option::is_none")]
pub speaker: Option<u32>,
```

- [ ] **Step 2: Update all WordInfo construction sites**

Find all places that create `WordInfo` and add `speaker: None`. This preserves backward compatibility.

- [ ] **Step 3: Extend Ready message**

In `src/protocol/mod.rs`, add to `ServerMessage::Ready`:

```rust
/// Whether speaker diarization is available (model loaded).
#[serde(skip_serializing_if = "std::ops::Not::not")]
diarization: bool,
```

- [ ] **Step 4: Extend Configure message**

In `ClientMessage::Configure`, change to:

```rust
Configure {
    #[serde(default)]
    sample_rate: Option<u32>,
    /// Enable speaker diarization for this session.
    #[serde(default)]
    diarization: Option<bool>,
},
```

Make `sample_rate` optional (Some = set rate, None = keep default). This allows Configure to set either or both.

- [ ] **Step 5: Fix all Ready and Configure construction/matching sites**

Update server to include `diarization: false` (or true when model loaded) in Ready.
Update Configure handler to accept both fields.

- [ ] **Step 6: Add tests**

```rust
#[test]
fn test_word_info_speaker_none_omitted() {
    let w = WordInfo { word: "test".into(), start: 0.0, end: 0.5, confidence: 0.9, speaker: None };
    let json = serde_json::to_string(&w).unwrap();
    assert!(!json.contains("speaker"));
}

#[test]
fn test_word_info_speaker_present() {
    let w = WordInfo { word: "test".into(), start: 0.0, end: 0.5, confidence: 0.9, speaker: Some(1) };
    let json = serde_json::to_string(&w).unwrap();
    assert!(json.contains("\"speaker\":1"));
}
```

- [ ] **Step 7: Run all tests**

Run: `cargo test --lib --features coreml,diarization`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/inference/mod.rs src/protocol/mod.rs
git commit -m "feat(protocol): add speaker field to WordInfo, diarization in Ready/Configure"
```

---

### Task 5: Integrate diarization into streaming pipeline

**Files:**
- Modify: `src/inference/mod.rs` (StreamingState + diarization state)
- Modify: `src/server/mod.rs` (load speaker model, handle diarization Configure)

- [ ] **Step 1: Add DiarizationState to StreamingState**

In `StreamingState`, add:

```rust
#[cfg(feature = "diarization")]
pub diarization: Option<DiarizationStreamState>,
```

Where `DiarizationStreamState` contains the audio buffer for embedding segments and the cluster:

```rust
#[cfg(feature = "diarization")]
pub struct DiarizationStreamState {
    pub audio_buffer: Vec<f32>,
    pub cluster: diarization::SpeakerCluster,
    pub current_speaker: Option<u32>,
}
```

- [ ] **Step 2: Run speaker embedding periodically during process_chunk**

After the existing RNN-T decode in `process_chunk`, if diarization is enabled:
1. Accumulate 16kHz audio into `diarization.audio_buffer`
2. When buffer reaches ~1.5s (24000 samples), extract embedding
3. Assign speaker ID via cluster
4. Attach `speaker: Some(id)` to all words in the current segment
5. Clear the diarization audio buffer

- [ ] **Step 3: Load speaker model in server on startup**

In server, conditionally load the speaker model:

```rust
#[cfg(feature = "diarization")]
let speaker_encoder = {
    let dir = std::path::Path::new(model_dir);
    match crate::inference::diarization::SpeakerEncoder::load(dir) {
        Ok(enc) => {
            tracing::info!("Speaker diarization model loaded");
            Some(Arc::new(enc))
        }
        Err(e) => {
            tracing::warn!("Speaker model not available: {e}. Diarization disabled.");
            None
        }
    }
};
```

Store in AppState and pass to WebSocket handler.

- [ ] **Step 4: Handle Configure { diarization: true }**

When client sends `Configure { diarization: Some(true) }`, enable diarization for the connection (create `DiarizationStreamState`). Only if speaker model is loaded.

- [ ] **Step 5: Run tests**

Run: `cargo test --lib --features coreml,diarization`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/inference/mod.rs src/server/mod.rs src/server/http.rs
git commit -m "feat(diarization): integrate speaker identification into streaming pipeline"
```

---

### Task 6: Tests and documentation

**Files:**
- Modify: `tests/server_integration.rs`
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add diarization protocol test**

In integration tests, add test for Configure { diarization: true } with mock/unavailable model.

- [ ] **Step 2: Update README.md**

Add speaker diarization section:
- Feature flag: `cargo build --features coreml,diarization`
- Model download: `gigastt download --diarization`
- Protocol: Configure { diarization: true }, speaker field in words
- Example JSON output with speaker IDs

- [ ] **Step 3: Update CLAUDE.md**

Add diarization module to architecture, feature flag docs.

- [ ] **Step 4: Commit**

```bash
git add tests/ README.md CLAUDE.md
git commit -m "docs: document speaker diarization feature and protocol"
```
