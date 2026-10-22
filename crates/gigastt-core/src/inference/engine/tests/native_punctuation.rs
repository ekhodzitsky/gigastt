//! Local real-model adapter/engine contracts; never silently skip missing weights.
use super::*;

#[test]
#[ignore = "requires GIGASTT_NATIVE_PUNCT_MODEL with local rupunct weights"]
fn test_native_punctuation_request_controls_and_lexical_preservation() -> anyhow::Result<()> {
    let path = std::env::var("GIGASTT_NATIVE_PUNCT_MODEL")?;
    let model = crate::punctuation::NativePunctuator::load(std::path::Path::new(&path))?;
    let oversized = "а".repeat(4 * 1024 * 1024 + 1);
    assert!(model.restore(&oversized) == oversized);
    assert_eq!(model.failed_restores(), 1);
    let spaced = model.restore("привет\nкак\tдела");
    assert!(spaced.contains('\n') && spaced.contains('\t'));
    let (engine, _temp) = crate::test_support::rnnt_engine();
    let engine = engine.with_native_punctuator(Some(model));
    assert!(engine.has_punctuator());
    let input = "привет как дела";
    let output =
        engine.apply_text_postprocess(input.into(), false, true, DecodeControls::default())?;
    let lexical = |text: &str| {
        text.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    assert_eq!(lexical(&output), lexical(input));
    assert_ne!(output, input, "real model must have run");
    assert_eq!(
        engine.apply_text_postprocess(input.into(), false, false, DecodeControls::default())?,
        input
    );
    let numeric = engine.apply_text_postprocess(
        "двадцать один".into(),
        true,
        true,
        DecodeControls::default(),
    )?;
    assert!(numeric.contains("21") && !numeric.contains("двадцать"));
    let cancelled = || true;
    let result = engine.apply_text_postprocess(
        input.into(),
        false,
        true,
        DecodeControls {
            abort: Some(&cancelled),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(crate::error::GigasttError::Cancelled)));
    let engine = engine.with_punctuator(None);
    assert!(!engine.has_punctuator());
    let engine = engine.with_native_punctuator(None);
    assert!(!engine.has_punctuator());
    Ok(())
}
