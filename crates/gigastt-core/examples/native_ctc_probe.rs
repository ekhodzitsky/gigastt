//! Diagnostic native-runtime probe, not a production model-loading path.
//! Usage: native_ctc_probe MODEL MANIFEST_JSON OUTPUT_DIR cpu|production THREADS
//! Manifest: [{"id":"case", "input":"raw.f32", "mode":"mel"|"pcm"}].
//! Mel files are little-endian f32 in contiguous [1, 64, T] layout.

use anyhow::{Context, Result, ensure};
use gigastt_core::inference::{FeatureExtractor, N_MELS};
use gigastt_core::runtime_api::{
    Shape, Tensor, TensorData, TensorDataView, cpu_factory, production_factory,
};
use serde::Deserialize;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Case {
    id: String,
    input: PathBuf,
    mode: String,
    feature_length: Option<i64>,
}

fn read_floats(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len().is_multiple_of(4),
        "input is not whole f32 samples"
    );
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|x| x.is_finite()), "non-finite input");
    Ok(values)
}

fn write_floats(path: &Path, values: &[f32]) -> Result<()> {
    let mut file = BufWriter::new(std::fs::File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("missing model")?);
    let manifest = PathBuf::from(args.next().context("missing manifest")?);
    let output = PathBuf::from(args.next().context("missing output directory")?);
    let factory_mode = args.next().context("missing factory mode")?;
    let threads: usize = args.next().context("missing thread count")?.parse()?;
    ensure!(
        (1..=8).contains(&threads),
        "diagnostic threads must be 1..8"
    );
    ensure!(args.next().is_none(), "unexpected argument");
    let factory = match factory_mode.as_str() {
        "cpu" => cpu_factory(),
        "production" => production_factory(model.parent().context("model needs parent directory")?),
        _ => anyhow::bail!("factory must be cpu or production"),
    };
    let runtime = factory.create(threads)?;
    let session = runtime.load_session(&model, true)?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(manifest)?)?;
    std::fs::create_dir_all(&output)?;
    for case in cases {
        ensure!(
            !case.id.is_empty()
                && case
                    .id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-')),
            "invalid output case id"
        );
        let input = read_floats(&case.input)?;
        let (features, frames) = match case.mode.as_str() {
            "mel" => {
                ensure!(
                    !input.is_empty() && input.len().is_multiple_of(N_MELS),
                    "invalid mel shape"
                );
                let frames = input.len() / N_MELS;
                (input, frames)
            }
            "pcm" => {
                ensure!(!input.is_empty(), "empty PCM input");
                FeatureExtractor::new().compute(&input)
            }
            _ => anyhow::bail!("mode must be mel or pcm"),
        };
        let frame_count = i64::try_from(frames).context("frame count exceeds i64")?;
        let feature_length = case.feature_length.unwrap_or(frame_count);
        ensure!(
            (1..=frame_count).contains(&feature_length),
            "invalid feature length"
        );
        write_floats(&output.join(format!("{}.features.f32", case.id)), &features)?;
        let inputs = [
            Tensor::new(
                Shape::new(vec![1, N_MELS, frames]),
                TensorData::F32(features),
            )?,
            Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![feature_length]))?,
        ];
        let results = session.run(&inputs)?;
        let logits = results.first().context("missing logits")?;
        let logits_view = logits.view();
        let logits_data = logits_view.data().as_f32().context("logits are not f32")?;
        write_floats(&output.join(format!("{}.logits.f32", case.id)), logits_data)?;
        let lengths = results.get(1).context("missing output lengths")?;
        let output_lengths = match lengths.view().data() {
            TensorDataView::I64(v) => v.to_vec(),
            TensorDataView::I32(v) => v.iter().map(|n| i64::from(*n)).collect(),
            _ => anyhow::bail!("output lengths are not integers"),
        };
        std::fs::write(
            output.join(format!("{}.json", case.id)),
            serde_json::to_vec_pretty(&serde_json::json!({
                "id": case.id, "model": model, "input": case.input,
                "input_mode": case.mode, "factory": factory_mode,
                "intra_threads": threads, "inter_threads": 1,
                "feature_shape": [1, N_MELS, frames], "feature_length": feature_length,
                "logit_shape": logits.shape().dims(), "output_lengths": output_lengths,
                "ort_build_info": ort::info(), "encoding": "little-endian f32",
            }))?,
        )?;
    }
    Ok(())
}
