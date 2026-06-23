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
/// Branches on the candle dtype to preserve element type (mirrors `to_candle`'s
/// dtype switch): F32 -> [`TensorData::F32`], I64 -> [`TensorData::I64`]. I32 is
/// not produced here because `to_candle` widens I32 inputs to I64, so they
/// round-trip as I64. Any other candle dtype (U8/U32/F16/BF16/F64) is never
/// produced by this backend and returns an error rather than silently casting.
pub(crate) fn from_candle(c: &CandleTensor) -> Result<Tensor, RuntimeError> {
    let dims = c.dims().to_vec();
    let flat = c.flatten_all().map_err(backend_err)?;
    match c.dtype() {
        DType::F32 => {
            let data = flat.to_vec1::<f32>().map_err(backend_err)?;
            Tensor::new(Shape::new(dims), TensorData::F32(data))
        }
        DType::I64 => {
            let data = flat.to_vec1::<i64>().map_err(backend_err)?;
            Tensor::new(Shape::new(dims), TensorData::I64(data))
        }
        other => Err(RuntimeError::InferenceFailed(format!(
            "from_candle: unsupported candle dtype {other:?}"
        ))),
    }
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

    #[test]
    fn test_i64_roundtrip_preserves_dtype_and_data() {
        let original = Tensor::new(Shape::new(vec![1, 3]), TensorData::I64(vec![1, 2, 3])).unwrap();

        let c = to_candle(&original, &Device::Cpu).unwrap();
        assert_eq!(c.dtype(), DType::I64);

        let recovered = from_candle(&c).unwrap();
        assert_eq!(recovered.shape().dims(), &[1, 3]);
        assert_eq!(recovered.view().data().as_i64(), Some(&[1, 2, 3][..]));
    }
}
