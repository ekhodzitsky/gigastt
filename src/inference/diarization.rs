//! Speaker diarization: embedding extraction and online incremental clustering.
//!
//! This module provides [`SpeakerEncoder`] for ONNX-based speaker embedding
//! extraction and [`SpeakerCluster`] for online incremental centroid clustering.

use anyhow::Context;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Mutex;

/// Dimension of speaker embedding vectors.
pub const EMBEDDING_DIM: usize = 256;

/// Number of audio samples per segment (1.5 s at 16 kHz).
pub const SEGMENT_SAMPLES: usize = 24000;

/// Speaker encoder that extracts fixed-size embeddings from audio segments.
///
/// Wraps a WeSpeaker ResNet34 ONNX session. Embeddings are L2-normalised so
/// cosine similarity equals the dot product.
///
/// Thread-safe: the ONNX session is wrapped in a [`Mutex`].
pub struct SpeakerEncoder {
    session: Mutex<Session>,
}

impl SpeakerEncoder {
    /// Load the speaker encoder from `model_dir/wespeaker_resnet34.onnx`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file is missing or the ONNX session
    /// cannot be created.
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let path = model_dir.join("wespeaker_resnet34.onnx");
        if !path.exists() {
            anyhow::bail!(
                "wespeaker_resnet34.onnx not found in {}",
                model_dir.display()
            );
        }
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&path)
            .context("Failed to load speaker encoder model")?;
        Ok(Self { session: Mutex::new(session) })
    }

    /// Extract an L2-normalised speaker embedding from raw 16 kHz f32 samples.
    ///
    /// The input is padded with zeros or truncated to exactly [`SEGMENT_SAMPLES`]
    /// before inference. The returned array has length [`EMBEDDING_DIM`].
    ///
    /// # Errors
    ///
    /// Returns an error if ONNX inference fails or the output tensor cannot be
    /// extracted.
    pub fn extract_embedding(&self, samples: &[f32]) -> anyhow::Result<[f32; EMBEDDING_DIM]> {
        // Pad or truncate to exactly SEGMENT_SAMPLES
        let mut buf = [0.0f32; SEGMENT_SAMPLES];
        let copy_len = samples.len().min(SEGMENT_SAMPLES);
        buf[..copy_len].copy_from_slice(&samples[..copy_len]);

        let input_tensor =
            TensorRef::from_array_view(([1_usize, SEGMENT_SAMPLES], buf.as_slice()))?;

        let mut session = self.session.lock().unwrap_or_else(|e| {
            tracing::warn!("SpeakerEncoder session mutex was poisoned, recovering");
            e.into_inner()
        });
        let outputs = session
            .run(ort::inputs![input_tensor])
            .context("SpeakerEncoder inference failed")?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract speaker embedding tensor")?;

        anyhow::ensure!(
            data.len() >= EMBEDDING_DIM,
            "Expected embedding dim >= {}, got {}",
            EMBEDDING_DIM,
            data.len()
        );

        let mut embedding = [0.0f32; EMBEDDING_DIM];
        embedding.copy_from_slice(&data[..EMBEDDING_DIM]);

        // L2 normalisation
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        Ok(embedding)
    }
}

/// Cosine similarity threshold for assigning an embedding to an existing speaker.
///
/// Embeddings with similarity above this threshold are merged into the nearest centroid;
/// below it, a new speaker identity is created.
const COSINE_THRESHOLD: f32 = 0.5;

/// Maximum number of distinct speaker identities tracked by [`SpeakerCluster`].
///
/// Once the limit is reached, new embeddings are assigned to the closest existing
/// centroid regardless of the similarity threshold.
const MAX_SPEAKERS: usize = 64;

/// Online incremental speaker clustering.
///
/// Maintains a set of speaker centroids and assigns incoming embeddings to the
/// nearest centroid (if above the configured threshold) or creates a new speaker.
/// Centroids are updated via running average after each assignment.
pub struct SpeakerCluster {
    centroids: Vec<[f32; EMBEDDING_DIM]>,
    counts: Vec<usize>,
    threshold: f32,
}

impl SpeakerCluster {
    /// Create an empty cluster with no known speakers using the default threshold.
    pub fn new() -> Self {
        Self::with_threshold(COSINE_THRESHOLD)
    }

    /// Create an empty cluster with a custom cosine similarity threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self { centroids: Vec::new(), counts: Vec::new(), threshold }
    }

    /// Assign an embedding to a speaker ID.
    ///
    /// Computes cosine similarity against all known centroids. If the best match
    /// exceeds the configured threshold, the centroid is updated via running average
    /// and the corresponding speaker ID is returned. Otherwise, a new speaker is
    /// registered and its ID (index) is returned.
    ///
    /// When [`MAX_SPEAKERS`] is reached, the embedding is always assigned to the
    /// closest existing centroid.
    pub fn assign(&mut self, embedding: &[f32; EMBEDDING_DIM]) -> u32 {
        let mut best_id: Option<usize> = None;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, centroid);
            if sim > best_sim {
                best_sim = sim;
                best_id = Some(i);
            }
        }

        // At speaker limit — assign to closest centroid regardless of threshold
        if self.centroids.len() >= MAX_SPEAKERS {
            let id = best_id.unwrap_or(0);
            let n = self.counts[id] as f32;
            let centroid = &mut self.centroids[id];
            for (c, &e) in centroid.iter_mut().zip(embedding.iter()) {
                *c = (*c * n + e) / (n + 1.0);
            }
            // Re-normalize centroid to prevent drift
            let norm: f32 = self.centroids[id].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut self.centroids[id] {
                    *x /= norm;
                }
            }
            self.counts[id] += 1;
            return id as u32;
        }

        if let Some(id) = best_id
            && best_sim > self.threshold
        {
            // Update centroid via running average
            let n = self.counts[id] as f32;
            let centroid = &mut self.centroids[id];
            for (c, &e) in centroid.iter_mut().zip(embedding.iter()) {
                *c = (*c * n + e) / (n + 1.0);
            }
            self.counts[id] += 1;
            // Re-normalize centroid to prevent drift
            let norm: f32 = self.centroids[id].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut self.centroids[id] {
                    *x /= norm;
                }
            }
            return id as u32;
        }

        // New speaker
        self.centroids.push(*embedding);
        self.counts.push(1);
        (self.centroids.len() - 1) as u32
    }

    /// Return the number of distinct speakers seen so far.
    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }
}

impl Default for SpeakerCluster {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two fixed-size embedding vectors.
///
/// Returns a value in `[-1.0, 1.0]`. Returns `0.0` if either vector is zero.
pub fn cosine_similarity(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(value: f32) -> [f32; EMBEDDING_DIM] {
        [value; EMBEDDING_DIM]
    }

    fn make_unit_embedding(index: usize) -> [f32; EMBEDDING_DIM] {
        let mut emb = [0.0f32; EMBEDDING_DIM];
        emb[index] = 1.0;
        emb
    }

    // --- cosine_similarity ---

    #[test]
    fn test_cosine_similarity_identical() {
        let a = make_embedding(1.0);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = make_unit_embedding(0);
        let b = make_unit_embedding(1);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "expected ~0.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = make_embedding(1.0);
        let b = make_embedding(-1.0);
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5, "expected ~-1.0, got {sim}");
    }

    // --- SpeakerCluster ---

    #[test]
    fn test_cluster_new_speaker() {
        let mut cluster = SpeakerCluster::new();
        let emb = make_embedding(1.0);
        let id = cluster.assign(&emb);
        assert_eq!(id, 0, "first speaker should be ID 0");
        assert_eq!(cluster.num_speakers(), 1);
    }

    #[test]
    fn test_cluster_same_speaker() {
        let mut cluster = SpeakerCluster::new();
        let emb_a = make_embedding(1.0);
        // Slightly different but very similar embedding (sim close to 1.0)
        let mut emb_b = make_embedding(1.0);
        emb_b[0] = 1.001;

        let id1 = cluster.assign(&emb_a);
        let id2 = cluster.assign(&emb_b);
        assert_eq!(id1, id2, "similar embeddings should map to the same speaker");
        assert_eq!(cluster.num_speakers(), 1);
    }

    #[test]
    fn test_cluster_different_speakers() {
        let mut cluster = SpeakerCluster::new();
        // Orthogonal embeddings -> similarity = 0.0, below threshold
        let emb_a = make_unit_embedding(0);
        let emb_b = make_unit_embedding(1);

        let id1 = cluster.assign(&emb_a);
        let id2 = cluster.assign(&emb_b);
        assert_ne!(id1, id2, "orthogonal embeddings should be different speakers");
        assert_eq!(cluster.num_speakers(), 2);
    }

    #[test]
    fn test_cluster_three_speakers() {
        let mut cluster = SpeakerCluster::new();
        // Three mutually orthogonal embeddings -- each should be a new speaker
        let emb_a = make_unit_embedding(0);
        let emb_b = make_unit_embedding(1);
        let emb_c = make_unit_embedding(2);

        let id_a1 = cluster.assign(&emb_a);
        let id_b = cluster.assign(&emb_b);
        let id_c = cluster.assign(&emb_c);
        // Return to speaker A with an identical embedding
        let id_a2 = cluster.assign(&emb_a);

        assert_eq!(id_a1, 0);
        assert_eq!(id_b, 1);
        assert_eq!(id_c, 2);
        assert_eq!(id_a2, id_a1, "returning to speaker A should yield the same ID");
        assert_eq!(cluster.num_speakers(), 3);
    }

    // --- SpeakerEncoder constants ---

    #[test]
    fn test_embedding_dim_constant() {
        assert_eq!(EMBEDDING_DIM, 256);
    }

    #[test]
    fn test_segment_samples_constant() {
        // 1.5 s * 16000 Hz = 24000
        assert_eq!(SEGMENT_SAMPLES, 24000);
    }

    #[test]
    fn test_load_returns_error_for_missing_file() {
        let result = SpeakerEncoder::load(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        let err = result.err().unwrap();
        let msg = format!("{err}");
        assert!(msg.contains("wespeaker_resnet34.onnx"));
    }
}
