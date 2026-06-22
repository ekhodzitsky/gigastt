/// Owned, cheaply cloneable tensor value used by the runtime abstraction layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: TensorData,
}

/// Owned tensor storage for the supported element types.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

/// Zero-copy borrow of tensor storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorDataView<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl<'a> TensorDataView<'a> {
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        match self {
            TensorDataView::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<&'a [i32]> {
        match self {
            TensorDataView::I32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<&'a [i64]> {
        match self {
            TensorDataView::I64(v) => Some(v),
            _ => None,
        }
    }
}

/// Normalized tensor shape independent of any runtime backend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

/// Known tensor element types supported by the runtime abstraction.
///
/// Only types that have a corresponding [`TensorData`] variant are listed here;
/// back-ends that produce other ONNX types must convert or reject them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    F32,
    I32,
    I64,
}

impl Tensor {
    /// Creates a tensor, validating that `data` length matches `shape`.
    pub fn new(shape: Shape, data: TensorData) -> Result<Self, crate::runtime::RuntimeError> {
        let expected = shape.elements();
        let actual = data.len();
        if expected != actual {
            return Err(crate::runtime::RuntimeError::DataLengthMismatch {
                expected,
                got: actual,
            });
        }
        Ok(Self { shape, data })
    }

    /// Convenience constructor that panics on shape/data mismatch.
    ///
    /// Use only when dimensions are statically known; prefer [`Self::new`] for
    /// runtime-sized tensors.
    pub fn new_checked(shape: Shape, data: TensorData) -> Self {
        Self::new(shape, data).expect("tensor data length mismatch")
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn element_type(&self) -> ElementType {
        match &self.data {
            TensorData::F32(_) => ElementType::F32,
            TensorData::I32(_) => ElementType::I32,
            TensorData::I64(_) => ElementType::I64,
        }
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            shape: &self.shape,
            data: match &self.data {
                TensorData::F32(v) => TensorDataView::F32(v.as_slice()),
                TensorData::I32(v) => TensorDataView::I32(v.as_slice()),
                TensorData::I64(v) => TensorDataView::I64(v.as_slice()),
            },
        }
    }

    pub fn into_data(self) -> TensorData {
        self.data
    }

    /// Return a mutable view of the underlying f32 buffer, if this tensor is f32.
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.data {
            TensorData::F32(v) => Some(v.as_mut_slice()),
            _ => None,
        }
    }

    /// Return a mutable view of the underlying i32 buffer, if this tensor is i32.
    pub fn as_i32_mut(&mut self) -> Option<&mut [i32]> {
        match &mut self.data {
            TensorData::I32(v) => Some(v.as_mut_slice()),
            _ => None,
        }
    }

    /// Return a mutable view of the underlying i64 buffer, if this tensor is i64.
    pub fn as_i64_mut(&mut self) -> Option<&mut [i64]> {
        match &mut self.data {
            TensorData::I64(v) => Some(v.as_mut_slice()),
            _ => None,
        }
    }

    /// Resize the tensor to a new shape, reusing the existing storage.
    ///
    /// The buffer is resized to the new element count and zero-padded if it
    /// grows. The caller must update the data before use.
    pub fn resize_to(&mut self, shape: Shape) {
        let new_len = shape.elements();
        match &mut self.data {
            TensorData::F32(v) => v.resize(new_len, 0.0),
            TensorData::I32(v) => v.resize(new_len, 0),
            TensorData::I64(v) => v.resize(new_len, 0),
        }
        self.shape = shape;
    }
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed view of a tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorView<'a> {
    shape: &'a Shape,
    data: TensorDataView<'a>,
}

impl<'a> TensorView<'a> {
    pub fn shape(&self) -> &Shape {
        self.shape
    }

    pub fn data(&self) -> &TensorDataView<'a> {
        &self.data
    }
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_and_data_match() {
        let t = Tensor::new(Shape::new(vec![2, 3]), TensorData::F32(vec![0.0; 6])).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.element_type(), ElementType::F32);
    }

    #[test]
    fn test_tensor_rejects_mismatched_data() {
        let err = Tensor::new(Shape::new(vec![2, 3]), TensorData::F32(vec![0.0; 5])).unwrap_err();
        assert!(matches!(
            err,
            crate::runtime::RuntimeError::DataLengthMismatch {
                expected: 6,
                got: 5
            }
        ));
    }

    #[test]
    fn test_shape_elements() {
        assert_eq!(Shape::new(vec![2, 3, 4]).elements(), 24);
        assert_eq!(Shape::new(vec![]).elements(), 1);
    }

    #[test]
    fn test_tensor_view_f32() {
        let t = Tensor::new(
            Shape::new(vec![2, 2]),
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        let v = t.view();
        assert_eq!(v.shape().dims(), &[2, 2]);
        assert_eq!(v.data().as_f32(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn test_tensor_view_non_f32_returns_none() {
        let t = Tensor::new(Shape::new(vec![2]), TensorData::I32(vec![1, 2])).unwrap();
        let v = t.view();
        assert_eq!(v.data().as_f32(), None);
    }

    #[test]
    fn test_shape_elements_zero_dimension() {
        assert_eq!(Shape::new(vec![0]).elements(), 0);
        assert_eq!(Shape::new(vec![2, 0]).elements(), 0);
    }
}
