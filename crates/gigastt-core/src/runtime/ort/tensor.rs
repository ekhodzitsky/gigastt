use ort::session::SessionInputValue;
use ort::value::{TensorElementType, Value};

use crate::runtime::{
    error::RuntimeError,
    tensor::{Shape, Tensor, TensorData, TensorDataView},
};

impl Tensor {
    /// Converts this owned tensor into an `ort` value.
    pub fn into_ort_value(self) -> Result<Value, RuntimeError> {
        let shape: Vec<i64> = self.shape().dims().iter().map(|&d| d as i64).collect();
        match self.into_data() {
            TensorData::F32(data) => ort::value::Tensor::from_array((shape, data))
                .map(|t| t.into_dyn())
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string())),
            TensorData::I32(data) => ort::value::Tensor::from_array((shape, data))
                .map(|t| t.into_dyn())
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string())),
            TensorData::I64(data) => ort::value::Tensor::from_array((shape, data))
                .map(|t| t.into_dyn())
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string())),
        }
    }

    /// Returns a borrowed `ort` input value backed by this tensor's data.
    ///
    /// The returned `SessionInputValue` borrows from `self`; the caller must
    /// keep this tensor alive for the duration of the `run` call.
    pub fn as_ort_input(&self) -> Result<SessionInputValue<'_>, RuntimeError> {
        let shape: Vec<i64> = self.shape().dims().iter().map(|&d| d as i64).collect();
        match self.view().data() {
            TensorDataView::F32(data) => {
                let tensor_ref: ort::value::TensorRef<'_, f32> =
                    ort::value::TensorRef::from_array_view((shape, *data))
                        .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
                Ok(tensor_ref.into_dyn().into())
            }
            TensorDataView::I32(data) => {
                let tensor_ref: ort::value::TensorRef<'_, i32> =
                    ort::value::TensorRef::from_array_view((shape, *data))
                        .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
                Ok(tensor_ref.into_dyn().into())
            }
            TensorDataView::I64(data) => {
                let tensor_ref: ort::value::TensorRef<'_, i64> =
                    ort::value::TensorRef::from_array_view((shape, *data))
                        .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
                Ok(tensor_ref.into_dyn().into())
            }
        }
    }
}

/// Converts an `ort` tensor value into our owned tensor type.
pub fn value_to_tensor(value: Value) -> Result<Tensor, RuntimeError> {
    match *value.data_type() {
        TensorElementType::Float32 => {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
            Tensor::new(
                Shape::new(shape.iter().map(|&d| d as usize).collect()),
                TensorData::F32(data.to_vec()),
            )
        }
        TensorElementType::Int32 => {
            let (shape, data) = value
                .try_extract_tensor::<i32>()
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
            Tensor::new(
                Shape::new(shape.iter().map(|&d| d as usize).collect()),
                TensorData::I32(data.to_vec()),
            )
        }
        TensorElementType::Int64 => {
            let (shape, data) = value
                .try_extract_tensor::<i64>()
                .map_err(|e| RuntimeError::InferenceFailed(e.to_string()))?;
            Tensor::new(
                Shape::new(shape.iter().map(|&d| d as usize).collect()),
                TensorData::I64(data.to_vec()),
            )
        }
        other => Err(RuntimeError::InferenceFailed(format!(
            "unsupported element type: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ort_roundtrip_f32() {
        let tensor = Tensor::new(
            Shape::new(vec![2, 3]),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        let value = tensor.clone().into_ort_value().unwrap();
        let recovered = value_to_tensor(value).unwrap();
        assert_eq!(tensor, recovered);
    }

    #[test]
    fn test_tensor_ort_roundtrip_i32() {
        let tensor = Tensor::new(Shape::new(vec![3]), TensorData::I32(vec![1, 2, 3])).unwrap();
        let value = tensor.clone().into_ort_value().unwrap();
        let recovered = value_to_tensor(value).unwrap();
        assert_eq!(tensor, recovered);
    }

    #[test]
    fn test_tensor_ort_roundtrip_i64() {
        let tensor =
            Tensor::new(Shape::new(vec![2, 2]), TensorData::I64(vec![1, 2, 3, 4])).unwrap();
        let value = tensor.clone().into_ort_value().unwrap();
        let recovered = value_to_tensor(value).unwrap();
        assert_eq!(tensor, recovered);
    }
}
