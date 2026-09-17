//! Engine unit tests, split by concern.

pub(super) use super::*;
use crate::inference::WordInfo;

pub(super) fn word(text: &str, start: f64, end: f64) -> WordInfo {
    WordInfo::new(text, start, end, 1.0, None)
}

mod backends;
mod cancellation;
mod commit_policy;
mod load;
mod mock;
mod paths;
mod predictor_probe;
mod stream;
mod transcribe;
