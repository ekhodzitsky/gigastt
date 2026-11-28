use super::*;

#[test]
fn test_v4_grid_distinguishes_physical_and_valid_lengths() {
    let grid = V4FrameGrid::new(960, 641).unwrap();
    assert_eq!(grid.mel_frames(), 5);
    assert_eq!(grid.valid_mel_frames(), 3);
    assert_eq!(grid.encoder_frames(), 2);
    assert_eq!(grid.valid_encoder_frames(), 1);
    assert_eq!(grid.mel_frame_start_sample(4), Some(640));
    assert_eq!(grid.mel_frame_start_sample(5), None);
    assert_eq!(grid.frame_shift_seconds(), 641.0 / 16000.0);
    assert_eq!(grid.frame_bounds(0), Some((0.0, 641.0 / 16000.0)));
    assert_eq!(grid.frame_bounds(1), None);
    assert!(V4FrameGrid::new(640, 641).is_err());
    assert!(V4FrameGrid::new(640, 319).is_err());
}

#[test]
fn test_v4_grid_short_odd_and_subsampling_residues() {
    for (n, m, t) in [
        (320, 1, 1),
        (479, 1, 1),
        (480, 2, 1),
        (641, 3, 1),
        (800, 4, 1),
        (960, 5, 2),
        (16037, 99, 25),
    ] {
        let grid = V4FrameGrid::new(n, n).unwrap();
        assert_eq!((grid.mel_frames(), grid.valid_mel_frames()), (m, m));
        assert_eq!((grid.encoder_frames(), grid.valid_encoder_frames()), (t, t));
        assert_eq!(grid.frame_shift_seconds(), n as f64 / 16000.0 / t as f64);
        assert_eq!(
            grid.frame_bounds(t - 1).unwrap().1,
            t as f64 * grid.frame_shift_seconds()
        );
    }
    for n in [0, 1, 319] {
        assert!(V4FrameGrid::new(n, n).is_err());
    }
}

#[test]
fn test_v4_word_grouping_preserves_bare_boundary_frame_and_unknown_piece() {
    let grid = V4FrameGrid::new(16037, 16037).unwrap();
    let pieces = ["▁", "hello", "<unk>", "▁", "мир"];
    let words = group_words(pieces.into_iter(), &[0, 1, 2, 7, 8], &grid).unwrap();
    assert_eq!(words.len(), 2);
    let shift = grid.frame_shift_seconds();
    assert_eq!(
        words[0],
        V4WordTiming {
            text: "hello<unk>".into(),
            start: 0.0,
            end: 3.0 * shift
        }
    );
    assert_eq!(
        words[1],
        V4WordTiming {
            text: "мир".into(),
            start: 7.0 * shift,
            end: 9.0 * shift
        }
    );
    assert!(group_words(["▁x"].into_iter(), &[25], &grid).is_err());
    assert!(group_words(["a", "b"].into_iter(), &[2, 1], &grid).is_err());
    assert!(group_words(["a"].into_iter(), &[0, 1], &grid).is_err());
}

#[test]
fn test_v4_tokenizer_rejects_invalid_non_model_bytes() {
    assert!(V4Tokenizer::from_bytes(TranscriptStyle::Raw, b"not sentencepiece").is_err());
}

/// Explicit model-free tokenizer parity against the pinned author fixture run.
/// The generated references remain local; only the reusable test logic is tracked.
#[test]
#[ignore = "requires pinned local author tokenizer fixtures and verified v4 pack; no inference"]
fn test_v4_pinned_author_tokenizer_parity() {
    use crate::sha256::{Sha256, hex_lower};
    use serde_json::{Value, json};
    use std::{fs, path::Path};

    fn digest(bytes: &[u8]) -> String {
        let mut hash = Sha256::new();
        hash.update(bytes);
        hex_lower(&hash.finalize())
    }
    fn verified(path: &Path, expected: &str) -> Vec<u8> {
        let raw = fs::read(path).unwrap();
        assert_eq!(digest(&raw), expected, "oracle fixture hash mismatch");
        raw
    }
    fn number(value: &Value) -> usize {
        usize::try_from(value.as_u64().unwrap()).unwrap()
    }
    fn ids(value: &Value) -> Vec<usize> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|id| {
                // Signed -1 in the official fixture is outside the Rust usize
                // API domain; usize::MAX exercises the equivalent rejection.
                usize::try_from(id.as_i64().unwrap()).unwrap_or(usize::MAX)
            })
            .collect()
    }
    let root = std::env::var_os("GIGASTT_V4_REFERENCE_DIR").expect("local reference directory");
    let root = Path::new(&root).join("reference");
    let pack_dir = std::env::var_os("GIGASTT_QUALIFIED_V4_PACK_DIR").expect("verified local pack");
    let pack = QualifiedV4Pack::load(Path::new(&pack_dir)).unwrap();
    let token_manifest: Value = serde_json::from_slice(&verified(
        &root.join("tokenizer-fixtures.json"),
        "2ddcbaa48053604786a6726e05967a9be235c0948efad9cb08a29997c0a9697f",
    ))
    .unwrap();
    let styles = token_manifest["styles"].as_array().unwrap();
    assert_eq!(styles.len(), 2);
    let mut token_results = Vec::new();
    for (style, case) in [TranscriptStyle::Raw, TranscriptStyle::E2e]
        .into_iter()
        .zip(styles)
    {
        let t = V4Tokenizer::from_pack(&pack, style).unwrap();
        assert_eq!(t.style(), style);
        assert_eq!(t.piece_count(), 1024);
        assert_eq!(t.unknown_id(), 0);
        assert_eq!(t.blank_id(), 1024);
        assert!(t.id_to_piece(1024).is_err());
        assert!(t.id_to_piece(usize::MAX).is_err());
        let mut failures = Vec::new();
        let pieces = case["pieces"].as_array().unwrap();
        assert_eq!(pieces.len(), 1024);
        for p in pieces {
            let id = number(&p["id"]);
            let literal = p["piece"].as_str().unwrap();
            if t.id_to_piece(id).unwrap() != literal
                || t.piece_to_id(literal) != id
                || t.decode(&[id]).unwrap() != p["decoded"].as_str().unwrap()
            {
                failures.push(json!({"kind":"piece","id":id,"actual_piece":t.id_to_piece(id).unwrap(),"actual_decoded":t.decode(&[id]).unwrap(),"expected":p}));
            }
        }
        assert_eq!(t.piece_to_id("<definitely-not-a-piece>"), 0);
        let encodes = case["encode_cases"].as_array().unwrap();
        assert_eq!(encodes.len(), 27);
        for p in encodes {
            let actual = t.encode(p["text"].as_str().unwrap()).unwrap();
            let decoded = t.decode(&actual).unwrap();
            if actual != ids(&p["ids"]) || decoded != p["decoded"].as_str().unwrap() {
                failures.push(json!({"kind":"encode","actual_ids":actual,"actual_decoded":decoded,"expected":p}));
            }
        }
        for (key, count) in [("decode_cases", 10), ("context_decode_cases", 6144)] {
            let cases = case[key].as_array().unwrap();
            assert_eq!(cases.len(), count);
            for p in cases {
                let actual = t.decode(&ids(&p["ids"]));
                if p.get("error").is_some() {
                    if let Ok(actual) = actual {
                        failures.push(json!({"kind":key,"actual":actual,"expected":p}));
                    }
                } else if actual.as_deref().ok() != p["decoded"].as_str() {
                    failures.push(json!({"kind":key,"actual":actual.ok(),"expected":p}));
                }
            }
        }
        let timing = case["timing_cases"].as_array().unwrap();
        assert_eq!(timing.len(), 4);
        for p in timing {
            let n = number(&p["samples"]);
            let grid = V4FrameGrid::new(n, n).unwrap();
            let tokens = ids(&p["token_ids"]);
            let frames = ids(&p["token_frames"]);
            let actual = t.words(&tokens, &frames, &grid).unwrap();
            let expected = p["words"].as_array().unwrap();
            let pass = grid.valid_encoder_frames() == number(&p["encoded_len"])
                && (grid.frame_shift_seconds() - p["frame_shift_seconds"].as_f64().unwrap()).abs()
                    <= 1e-9
                && t.decode(&tokens).unwrap() == p["decoded"].as_str().unwrap()
                && actual.len() == expected.len()
                && actual.iter().zip(expected).all(|(a, e)| {
                    a.text == e["text"].as_str().unwrap()
                        && (a.start - e["start"].as_f64().unwrap()).abs() <= 1e-9
                        && (a.end - e["end"].as_f64().unwrap()).abs() <= 1e-9
                });
            if !pass {
                failures.push(json!({"kind":"timing","actual_words":actual.iter().map(|w|json!({"text":w.text,"start":w.start,"end":w.end})).collect::<Vec<_>>(),"expected":p}));
            }
        }
        token_results.push(json!({"style":case["style"],"pieces":pieces.len(),"encodes":encodes.len(),"decode_cases":10,"context_decode_cases":6144,"timing_cases":timing.len(),"pass":failures.is_empty(),"failures":failures}));
    }
    println!(
        "V4_TOKEN_PARITY {}",
        serde_json::to_string(&token_results).unwrap()
    );
    assert!(token_results.iter().all(|x| x["pass"] == true));
}
