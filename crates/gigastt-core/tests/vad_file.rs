//! Integration test: VAD file path must skip silence without dropping speech.
//!
//! Model-gated (`#[ignore]`): requires the GigaAM model (~850 MB) at
//! `~/.gigastt/models` AND the Silero VAD model at
//! `~/.gigastt/models/vad/silero_vad.onnx`. Run with:
//! `cargo test -p gigastt-core --test vad_file -- --ignored --nocapture`.
//!
//! Guards the two correctness invariants of the silence-skipping file path:
//! (1) decoding only the VAD speech regions yields essentially the same words
//! as decoding the whole clip (VAD must not eat speech), and (2) the remapped
//! word timestamps stay on the original timeline (no compression artifacts).

use std::collections::HashSet;
use std::path::Path;

use gigastt_core::inference::Engine;
use gigastt_core::model::{default_model_dir, default_vad_model_dir};
use gigastt_core::vad::{SileroVad, VAD_MODEL_FILE, VadConfig};

fn norm_words(s: &str) -> HashSet<String> {
    s.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

#[test]
#[ignore = "requires the GigaAM model (~850MB) + Silero VAD model"]
fn vad_file_skips_silence_keeps_transcript() {
    let model_dir = default_model_dir();
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gigastt/tests/fixtures/golos_00.wav"
    );

    // Reference: decode the whole clip (no VAD).
    let plain = Engine::load(&model_dir).expect("load plain engine");
    let mut t = plain.pool.checkout_blocking().expect("checkout");
    let plain_res = plain
        .transcribe_file(fixture, &mut t)
        .expect("plain transcribe");
    drop(t);

    // VAD: decode only detected speech regions, remapping timestamps back.
    // The Silero VAD model is an optional separate download; skip gracefully if
    // it is absent (only the GigaAM model is required to reach this point).
    let vad_path = Path::new(&default_vad_model_dir()).join(VAD_MODEL_FILE);
    if !vad_path.exists() {
        eprintln!(
            "skipping vad_file: Silero VAD model not present at {}",
            vad_path.display()
        );
        return;
    }
    let vad = SileroVad::load(&vad_path).expect("load silero vad");
    let vad_engine = Engine::load(&model_dir)
        .expect("load vad engine")
        .with_vad(Some(vad), VadConfig::default());
    let mut t2 = vad_engine.pool.checkout_blocking().expect("checkout vad");
    let vad_res = vad_engine
        .transcribe_file(fixture, &mut t2)
        .expect("vad transcribe");

    eprintln!("plain: {:?}", plain_res.text);
    eprintln!("vad:   {:?}", vad_res.text);

    let pw = norm_words(&plain_res.text);
    let vw = norm_words(&vad_res.text);
    assert!(
        !vw.is_empty(),
        "VAD transcript is empty: {:?}",
        vad_res.text
    );

    // VAD must not drop content words: the silence-skipped transcript shares
    // (almost) all words with the full decode.
    let shared = pw.intersection(&vw).count();
    let overlap = if pw.is_empty() {
        1.0
    } else {
        shared as f64 / pw.len() as f64
    };
    assert!(
        overlap >= 0.8,
        "VAD transcript diverges from full decode: word-overlap {overlap:.2} (< 0.80)\n  \
         vad:   {:?}\n  plain: {:?}",
        vad_res.text,
        plain_res.text
    );

    // Remapped word timestamps must stay on the original timeline (no word
    // lands past the clip duration; a remap bug would compress them).
    let dur = vad_res.duration_s;
    for w in &vad_res.words {
        assert!(
            w.start >= 0.0 && w.end <= dur + 0.5,
            "word {:?} [{:.2},{:.2}] outside clip duration {dur:.2}s — timestamp remap regressed",
            w.word,
            w.start,
            w.end
        );
    }
}
