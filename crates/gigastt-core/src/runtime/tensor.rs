/// Owned, cheaply cloneable tensor value used by the runtime abstraction layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: TensorData,
}

/// Owned tensor storage for the supported element types.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

/// Zero-copy borrow of tensor storage.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub enum TensorDataView<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

#[allow(dead_code)]
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

/// Supported tensor element types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ElementType {
    F32,
    I32,
    I64,
}

#[allow(dead_code)]
impl Tensor {
    pub fn new(shape: Shape, data: TensorData) -> Self {
        let expected = shape.elements();
        let actual = data.len();
        assert_eq!(
            expected, actual,
            "tensor data length mismatch: expected {expected}, got {actual}"
        );
        Self { shape, data }
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
            shape: self.shape.clone(),
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
}

#[allow(dead_code)]
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
pub struct TensorView<'a> {
    shape: Shape,
    data: TensorDataView<'a>,
}

#[allow(dead_code)]
impl<'a> TensorView<'a> {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn data(&self) -> &TensorDataView<'a> {
        &self.data
    }
}

#[allow(dead_code)]
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
        let t = Tensor::new(Shape::new(vec![2, 3]), TensorData::F32(vec![0.0; 6]));
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.element_type(), ElementType::F32);
    }

    #[test]
    #[should_panic(expected = "tensor data length mismatch")]
    fn test_tensor_rejects_mismatched_data() {
        Tensor::new(Shape::new(vec![2, 3]), TensorData::F32(vec![0.0; 5]));
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
        );
        let v = t.view();
        assert_eq!(v.shape().dims(), &[2, 2]);
        assert_eq!(v.data().as_f32(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn test_tensor_view_non_f32_returns_none() {
        let t = Tensor::new(Shape::new(vec![2]), TensorData::I32(vec![1, 2]));
        let v = t.view();
        assert_eq!(v.data().as_f32(), None);
    }

    #[test]
    fn test_shape_elements_zero_dimension() {
        assert_eq!(Shape::new(vec![0]).elements(), 0);
        assert_eq!(Shape::new(vec![2, 0]).elements(), 0);
    }
}
