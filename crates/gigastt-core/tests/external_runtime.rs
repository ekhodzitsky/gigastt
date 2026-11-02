//! The runtime extension point must be usable by an external adapter crate.

use gigastt_core::runtime_api::{Runtime, RuntimeError, RuntimeFactory};
use gigastt_core::{inference::Engine, model::ModelVariant};

struct ExternalFactory;

impl RuntimeFactory for ExternalFactory {
    fn create(&self, _threads: usize) -> Result<Box<dyn Runtime>, RuntimeError> {
        Err(RuntimeError::InferenceFailed(
            "external factory was reached".into(),
        ))
    }

    fn cpu_fallback(&self) -> Box<dyn RuntimeFactory> {
        Box::new(Self)
    }
}

#[test]
fn test_external_runtime_factory_is_used_by_engine_loader() {
    let dir = tempfile::tempdir().unwrap();
    let result = Engine::load_with_factory(
        dir.path(),
        Some(ModelVariant::Rnnt),
        1,
        1,
        0,
        Box::new(ExternalFactory),
        1,
        false,
        "external",
    );
    let error = match result {
        Ok(_) => panic!("factory should fail"),
        Err(error) => error,
    };
    assert!(format!("{error:?}").contains("external factory was reached"));
}
