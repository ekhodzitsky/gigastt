//! `impl Engine` methods — split out of the former god-file.
use super::*;
impl Engine {
    /// Load ONNX models from the given directory and create an inference engine.
    ///
    /// Creates a pool of `DEFAULT_POOL_SIZE` session triplets for concurrent inference.
    /// The recognition head ([`ModelVariant`]) is auto-detected from the encoder
    /// file present on disk: `v3_rnnt_encoder.onnx` (or `_int8.onnx`) selects the
    /// plain rnnt head, else `v3_e2e_rnnt_encoder.onnx` selects e2e_rnnt. The
    /// matching decoder, joiner, and vocab files must also be present.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::ModelLoad`] if model files are missing or ONNX session creation fails.
    pub fn load(model_dir: &str) -> Result<Self, GigasttError> {
        Self::load_with_pool_size(model_dir, DEFAULT_POOL_SIZE)
    }

    /// Load ONNX models with a custom pool size. Requires the *full* pool to
    /// load (every triplet); use [`Engine::load_with_pool_size_min`] to boot on
    /// a partial pool.
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> Result<Self, GigasttError> {
        Self::load_with_pool_size_min(model_dir, pool_size, pool_size)
    }

    /// Load ONNX models with a custom pool size, tolerating a partial pool down
    /// to `min_size` triplets when the rest fail to load. Boots a degraded pool
    /// with a warning when `min_size <= loaded < pool_size`; errors only when
    /// fewer than `min_size` triplets load. `min_size` is clamped to
    /// `1..=pool_size`.
    pub fn load_with_pool_size_min(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
    ) -> Result<Self, GigasttError> {
        // No batch/stream split: the whole pool is interactive.
        Self::load_with_pools(model_dir, pool_size, min_size, 0)
    }

    /// Load ONNX models splitting the pool into an interactive pool (WebSocket +
    /// SSE) and a dedicated batch pool of `batch_pool_size` triplets for REST
    /// file transcription, so a long batch job can't starve real-time
    /// streaming. `batch_pool_size == 0` disables the split (REST shares the
    /// interactive pool); it is clamped to leave at least one interactive
    /// triplet. Partial-load tolerance follows `min_size` as in
    /// [`Engine::load_with_pool_size_min`].
    pub fn load_with_pools(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
    ) -> Result<Self, GigasttError> {
        Self::load_with_pools_threads(model_dir, pool_size, min_size, batch_pool_size, 1)
    }

    /// Like [`Engine::load_with_pools`], but with a configurable encoder
    /// intra-op thread count for the CPU EP. `encoder_intra_threads == 1` (the
    /// default everywhere else) builds sessions identical to the prior
    /// behaviour. Values `> 1` give the dominant encoder more intra-op
    /// parallelism on weak CPUs / long single-file jobs; the count is clamped
    /// against the logical CPU count so `pool_size * threads` can't oversubscribe
    /// the machine (see `Engine::clamp_encoder_intra_threads`). Ignored by the
    /// CoreML / CUDA builds (the accelerator owns scheduling there).
    pub fn load_with_pools_threads(
        model_dir: &str,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        Self::load_with_pools_threads_variant(
            model_dir,
            None,
            pool_size,
            min_size,
            batch_pool_size,
            encoder_intra_threads,
        )
    }

    /// Like [`Engine::load_with_pools_threads`], but lets the caller force which
    /// recognition head to load instead of auto-detecting it.
    ///
    /// When `variant` is `Some(v)`, head `v` is loaded (and the load fails with a
    /// clear `ModelLoad` error if `v`'s files aren't in `model_dir`). When it is
    /// `None`, the on-disk layout is auto-detected with `rnnt` precedence, exactly
    /// as [`Engine::load_with_pools_threads`]. This is the entry point that makes
    /// `--model-variant` effective when a directory holds more than one head.
    pub fn load_with_pools_threads_variant(
        model_dir: &str,
        variant: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        Self::load_with_pools_threads_variant_cache(
            model_dir,
            variant,
            pool_size,
            min_size,
            batch_pool_size,
            encoder_intra_threads,
            None,
        )
    }

    /// Like [`Engine::load_with_pools_threads_variant`], with an optional
    /// override for the CPU encoder's ORT optimized-graph cache directory.
    /// `None` keeps the default `<model_dir>/optimized_cache`; `Some(dir)`
    /// relocates the cache (e.g. a writable cache directory when the model
    /// dir is read-only). Ignored by the CoreML / CUDA / candle builds.
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_pools_threads_variant_cache(
        model_dir: &str,
        variant: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        encoder_intra_threads: usize,
        optimized_cache_dir: Option<PathBuf>,
    ) -> Result<Self, GigasttError> {
        let dir = Path::new(model_dir);
        // Resolve the head once, up front: an explicit `variant` wins, else
        // manifest.toml architecture, else detect from disk (rnnt precedence).
        // Resolving here (not just inside `load_with_factory`) keeps the RAM cap
        // sized to the head that will actually load.
        let variant = resolve_variant_required(variant, dir)?;
        // Bound the idle footprint: each pooled triplet deserializes its own
        // encoder copy, so a large `--pool-size` on a small host can OOM at
        // load. Clamp by available RAM (logs when it clamps); a no-op on hosts
        // with ample memory.
        let encoder_bytes = std::fs::metadata(encoder_model_path(dir, variant))
            .map(|m| m.len())
            .unwrap_or(0);
        let pool_size =
            Self::cap_pool_size_for_ram(pool_size, encoder_bytes, sizing::effective_ram_bytes());
        // Don't let `pool_size * encoder_intra_threads` oversubscribe the CPU
        // (no-op when the default `1` is requested).
        let logical_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let encoder_intra_threads =
            Self::clamp_encoder_intra_threads(pool_size, encoder_intra_threads, logical_cpus);

        let factory =
            production_factory_variant_with_cache(dir, Some(variant), optimized_cache_dir);
        Self::load_with_factory(
            dir,
            Some(variant),
            pool_size,
            min_size,
            batch_pool_size,
            factory,
            encoder_intra_threads,
        )
    }

    /// Assemble an engine from already-loaded sessions, tokenizer, and variant.
    ///
    /// Optional attachments (punctuator, ITN, biaser, VAD) stay off — callers
    /// chain [`Engine::with_punctuator`] / [`Engine::with_itn`] / etc.
    pub(crate) fn from_loaded_parts(
        pool: SessionPool,
        batch_pool: Option<SessionPool>,
        tokenizer: Tokenizer,
        variant: ModelVariant,
        int8: bool,
        ane_encoder: bool,
        #[cfg(feature = "diarization")] speaker_encoder: Option<LazySpeakerEncoder>,
    ) -> Self {
        Self {
            pool,
            batch_pool,
            tokenizer,
            features: FeatureExtractor::new(),
            variant,
            punctuator: None,
            itn: false,
            biaser: None,
            vad: None,
            vad_config: crate::vad::VadConfig::default(),
            endpoint_mode: EndpointMode::Auto,
            stream_max_window_samples: STREAM_MAX_WINDOW_SAMPLES,
            // Default on: stable-prefix commits cut long-phrase streaming WER
            // loss at slide boundaries (10.6% → 2.1% corpus WER on 10 labelled
            // Golos fixtures at the default 2.5 s window) with no measurable
            // TTFP or encoder-cost change.
            stream_stable_prefix: true,
            int8,
            ane_encoder,
            file_window_concurrency: 1,
            #[cfg(feature = "diarization")]
            speaker_encoder,
        }
    }

    /// Package-private factory-based loader. Used by production code paths and
    /// by tests that inject a [`crate::runtime::factory::RuntimeFactory`].
    pub(crate) fn load_with_factory(
        model_dir: &Path,
        variant_override: Option<ModelVariant>,
        pool_size: usize,
        min_size: usize,
        batch_pool_size: usize,
        factory: Box<dyn crate::runtime::factory::RuntimeFactory>,
        encoder_intra_threads: usize,
    ) -> Result<Self, GigasttError> {
        // Honor an explicit variant (e.g. from `--model-variant`) when the caller
        // resolved one; otherwise prefer `manifest.toml`, else auto-detect from
        // the on-disk layout (rnnt precedence). Passing the override through is
        // what makes `--model-variant` effective when a directory holds more than
        // one head — without it the engine always re-detects and silently loads
        // the highest-precedence head.
        let variant = resolve_variant_required(variant_override, model_dir)?;
        let pool_size = pool_size.max(1);
        let min_size = min_size.clamp(1, pool_size);

        let runtime = factory.create(encoder_intra_threads)?;
        let model_load = |e: anyhow::Error| GigasttError::ModelLoad {
            path: model_dir.display().to_string(),
            source: Some(e.into()),
        };

        let files = ResolvedModelFiles::resolve(model_dir, variant).map_err(model_load)?;
        if factory.verify_on_disk_checksums() {
            files.verify_pinned_checksums(variant)?;
        }

        tracing::info!("Detected model variant: {variant:?}");
        // The candle backend loads FP32 `candle/*.safetensors` and ignores the
        // INT8 ONNX encoder, so report no INT8 there and gate out the INT8 / ONNX
        // / CPU-EP logs below (which are wrong for candle), emitting an accurate
        // candle line instead. The default (ort) logging is unchanged.
        let is_int8 = !cfg!(feature = "candle") && files.using_int8;
        if !cfg!(feature = "candle") {
            // ORT product path is INT8-only (`ResolvedModelFiles` rejects FP32).
            tracing::info!("Using INT8 quantized encoder");
            tracing::info!(
                "Loading ONNX models from {} (pool_size={pool_size})...",
                model_dir.display()
            );
        }

        #[cfg(feature = "candle")]
        tracing::info!(
            "Loading Candle models from {} (pool_size={pool_size})...",
            model_dir.display()
        );
        #[cfg(feature = "coreml")]
        tracing::info!("Using CoreML execution provider (Neural Engine + CPU)");
        #[cfg(feature = "cuda")]
        tracing::info!("Using CUDA execution provider (falls back to CPU if unavailable)");
        #[cfg(not(any(feature = "coreml", feature = "cuda", feature = "candle")))]
        tracing::info!("Using CPU execution provider");

        // CoreML can reject a model at load time; fall back to CPU if that happens.
        #[cfg(feature = "coreml")]
        let triplets = match load_triplets_runtime(&*runtime, &files, variant, pool_size, min_size)
            .map_err(model_load)
        {
            Ok(triplets) => triplets,
            Err(load_err) => {
                tracing::warn!(
                    "CoreML EP failed to load sessions ({load_err:#}); falling back to CPU execution provider"
                );
                let cpu_factory = factory.cpu_fallback();
                let runtime = cpu_factory.create(encoder_intra_threads)?;
                load_triplets_runtime(&*runtime, &files, variant, pool_size, min_size)
                    .map_err(model_load)?
            }
        };
        #[cfg(not(feature = "coreml"))]
        let triplets = load_triplets_runtime(&*runtime, &files, variant, pool_size, min_size)
            .map_err(model_load)?;

        let tokenizer = Tokenizer::load(&files.vocab).map_err(model_load)?;

        tracing::info!(
            "Models loaded (vocab_size={}, pool_size={pool_size})",
            tokenizer.vocab_size()
        );

        // Probe only — do not open the WeSpeaker ONNX session at boot. The
        // encoder is loaded on the first diarization request.
        #[cfg(feature = "diarization")]
        let speaker_encoder = diarization::probe_speaker_encoder(model_dir);

        // Detect ANE from the loaded encoder session (not compile-time alone) so
        // non-rnnt heads / injected factories keep the ort chunk window.
        let ane_encoder = triplets.first().is_some_and(|t| t.encoder.is_ane_encoder());
        let (pool, batch_pool) = Self::split_triplets(triplets, batch_pool_size);
        let engine = Self::from_loaded_parts(
            pool,
            batch_pool,
            tokenizer,
            variant,
            is_int8,
            ane_encoder,
            #[cfg(feature = "diarization")]
            speaker_encoder,
        );

        // CoreML compiles its graph partitions lazily, so sessions that loaded
        // fine can still fail at the first `Run()`. Probe one triplet now; if the
        // probe fails, rebuild the pool on the CPU EP.
        #[cfg(feature = "coreml")]
        let engine = sizing::probe_or_rebuild(
            engine,
            |e: &Self| e.warmup_one().map_err(anyhow::Error::from),
            |mut e, probe_err| {
                tracing::warn!(
                    "CoreML EP failed at runtime ({probe_err:#}); falling back to CPU execution provider"
                );
                let cpu_factory = factory.cpu_fallback();
                let runtime = cpu_factory
                    .create(encoder_intra_threads)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let triplets =
                    load_triplets_runtime(&*runtime, &files, variant, pool_size, min_size)?;
                let (pool, batch_pool) = Self::split_triplets(triplets, batch_pool_size);
                e.pool = pool;
                e.batch_pool = batch_pool;
                Ok(e)
            },
        )
        .map_err(model_load)?;

        Ok(engine)
    }

    /// Split loaded triplets into an interactive pool and an optional batch
    /// pool of `batch_pool_size` triplets. Always leaves at least one triplet
    /// for the interactive pool; `batch_pool_size == 0` (or a pool too small to
    /// split) yields no batch pool.
    pub(crate) fn split_triplets(
        triplets: Vec<SessionTriplet>,
        batch_pool_size: usize,
    ) -> (SessionPool, Option<SessionPool>) {
        Self::split_pool(triplets, batch_pool_size)
    }

    /// Generic pool split underlying [`Engine::split_triplets`]: partition
    /// `items` into an interactive pool and an optional batch pool of
    /// `batch_pool_size` items, always leaving at least one item interactive.
    /// `batch_pool_size == 0` (or too few items to split) yields no batch pool.
    /// Generic over the item type so the routing can be unit-tested with a
    /// synthetic `Pool<u32>` instead of model-backed `SessionTriplet`s.
    pub(crate) fn split_pool<T: Send>(
        items: Vec<T>,
        batch_pool_size: usize,
    ) -> (Pool<T>, Option<Pool<T>>) {
        let (interactive, batch) = sizing::split_pool_items(items, batch_pool_size);
        (Pool::new(interactive), batch.map(Pool::new))
    }

    /// See [`sizing::cap_pool_size_for_ram`].
    pub(crate) fn cap_pool_size_for_ram(
        requested: usize,
        encoder_bytes: u64,
        total_ram: u64,
    ) -> usize {
        sizing::cap_pool_size_for_ram(requested, encoder_bytes, total_ram)
    }

    /// See [`sizing::clamp_encoder_intra_threads`].
    pub(crate) fn clamp_encoder_intra_threads(
        pool_size: usize,
        requested: usize,
        logical_cpus: usize,
    ) -> usize {
        sizing::clamp_encoder_intra_threads(pool_size, requested, logical_cpus)
    }

    /// Run one ~1 s silent inference on a single pooled session triplet.
    ///
    /// Exercises the full mel + encoder + RNN-T decode pipeline, forcing
    /// lazy EP work (CoreML partition compilation, first-run allocations) to
    /// happen now instead of on the first real request — and doubling as a
    /// runtime self-check for EPs that can fail at prediction time even
    /// though their sessions loaded fine (issue #42).
    pub(crate) fn warmup_one(&self) -> Result<(), GigasttError> {
        self.warmup_one_on(&self.pool)
    }

    /// Warm a single triplet from a specific pool with ~1 s of silence.
    pub(crate) fn warmup_one_on(&self, pool: &SessionPool) -> Result<(), GigasttError> {
        let silence = vec![0.0f32; 16000]; // 1 s at 16 kHz
        let mut guard = pool
            .checkout_blocking()
            .map_err(|e| GigasttError::Inference {
                source: Box::new(e),
            })?;
        self.transcribe_samples(&silence, &mut guard)?;
        Ok(())
    }

    /// Warm up every pooled session triplet with a ~1 s silent inference
    /// so the first real request doesn't pay the EP compile /
    /// first-allocation cost.
    ///
    /// Sequential checkouts visit each pooled triplet exactly once because
    /// check-in returns items to the back of the FIFO queue.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::Inference`] if a warmup inference fails — with
    /// the `coreml` feature this is unexpected, because [`Engine::load`]
    /// already probed the pool and fell back to the CPU EP if needed.
    pub fn warmup(&self) -> Result<(), GigasttError> {
        for _ in 0..self.pool.total() {
            self.warmup_one()?;
        }
        if let Some(ref batch) = self.batch_pool {
            for _ in 0..batch.total() {
                self.warmup_one_on(batch)?;
            }
        }
        Ok(())
    }

    /// The pool REST file transcription should use: the dedicated batch pool
    /// when one was split off, otherwise the interactive pool.
    pub fn pool_for_batch(&self) -> &SessionPool {
        self.batch_pool.as_ref().unwrap_or(&self.pool)
    }

    /// Close both the interactive and batch pools so every waiter wakes with
    /// `PoolError::Closed` during graceful shutdown.
    pub fn close_pools(&self) {
        self.pool.close();
        if let Some(ref batch) = self.batch_pool {
            batch.close();
        }
    }
}
