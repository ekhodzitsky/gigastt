//! Bridge between the runtime-abstraction [`Tensor`] type and `candle_core::Tensor`.
//!
//! ISOLATION: `candle_core` is referenced only here and in the sibling candle
//! modules; the rest of the crate never sees a `candle_core::Tensor`.

use candle_core::{DType, Device, Tensor as CandleTensor};

use crate::runtime::{
    error::RuntimeError,
    tensor::{Shape, Tensor, TensorData, TensorDataView},
};

fn backend_err(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::InferenceFailed(e.to_string())
}

/// Convert a runtime [`Tensor`] into a `candle_core::Tensor` on `dev`.
///
/// Supports F32, I64, and I32 element types. The candle tensor preserves the
/// source shape; I32 is widened to I64 (candle has no native i32).
pub(crate) fn to_candle(t: &Tensor, dev: &Device) -> Result<CandleTensor, RuntimeError> {
    let dims = t.shape().dims().to_vec();
    match t.view().data() {
        TensorDataView::F32(data) => {
            CandleTensor::from_slice(data, dims.as_slice(), dev).map_err(backend_err)
        }
        TensorDataView::I64(data) => {
            CandleTensor::from_slice(data, dims.as_slice(), dev).map_err(backend_err)
        }
        TensorDataView::I32(data) => {
            // candle has no native i32; widen to i64 (lossless) so length-style
            // integer inputs round-trip without overflow.
            let widened: Vec<i64> = data.iter().map(|&v| v as i64).collect();
            CandleTensor::from_slice(&widened, dims.as_slice(), dev).map_err(backend_err)
        }
    }
}

/// Convert a `candle_core::Tensor` back into a runtime [`Tensor`].
///
/// The candle tensor is cast to F32 (the encoder output is f32) and flattened
/// to a contiguous `Vec<f32>`; its dims become the runtime [`Shape`].
pub(crate) fn from_candle(c: &CandleTensor) -> Result<Tensor, RuntimeError> {
    let dims = c.dims().to_vec();
    let data = c
        .to_dtype(DType::F32)
        .map_err(backend_err)?
        .flatten_all()
        .map_err(backend_err)?
        .to_vec1::<f32>()
        .map_err(backend_err)?;
    Tensor::new(Shape::new(dims), TensorData::F32(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_roundtrip_preserves_shape_and_data() {
        let original = Tensor::new(
            Shape::new(vec![1, 2, 3]),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();

        let dev = Device::Cpu;
        let c = to_candle(&original, &dev).unwrap();
        assert_eq!(c.dims(), &[1, 2, 3]);

        let recovered = from_candle(&c).unwrap();
        assert_eq!(recovered.shape().dims(), &[1, 2, 3]);
        assert_eq!(
            recovered.view().data().as_f32(),
            Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..])
        );
    }

    #[test]
    fn test_i64_to_candle_preserves_values() {
        let t = Tensor::new(Shape::new(vec![1]), TensorData::I64(vec![123])).unwrap();
        let c = to_candle(&t, &Device::Cpu).unwrap();
        assert_eq!(c.dims(), &[1]);
        assert_eq!(
            c.flatten_all().unwrap().to_vec1::<i64>().unwrap(),
            vec![123]
        );
    }
}
