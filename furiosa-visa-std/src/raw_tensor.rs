use ndarray::{ArrayD, Axis, IxDyn};
use num_traits::Zero;
use std::{fmt::Debug, marker::PhantomData};

use abi_stable::std_types::RResult;
use furiosa_mapping::*;

use super::scalar::*;

/// Generates axes from a mapping.
pub(crate) fn gen_axes<Mapping: M>() -> Vec<Term> {
    let mut index = Index::new();
    index.add_mapping::<Mapping>(0);
    index
        .finalize()
        .expect("Invalid mapping")
        .into_iter()
        .map(|(term, _)| term)
        .collect::<Vec<_>>()
}

/// Tensor with scalar type `D`.
#[derive(Debug, Clone, PartialEq)]
pub struct RawTensor<D: Scalar> {
    /// The axes of the tensor represented as a sorted vector of `Term`.
    pub(crate) axes: Vec<Term>,
    /// The multi-dimensional array holding the tensor data, where each element maybe be uninitialized.
    pub(crate) data: ArrayD<Opt<D>>,
    pub(crate) _marker: PhantomData<D>,
}

impl<D: Scalar> Eq for RawTensor<D> where D: Eq {}

impl<D: Scalar> RawTensor<D> {
    /// Similar to `from_vec`, but creates a new tensor with uninit elements.
    pub fn from_elem<Mapping: M>(elem: Opt<D>) -> Self {
        // Construct axes from mapping.
        let axes = gen_axes::<Mapping>();

        // Construct data array from vector.
        let shape = axes.iter().map(|term| term.modulo).collect::<Vec<usize>>();
        let data = ArrayD::from_elem(shape, elem);

        Self {
            axes,
            data,
            _marker: PhantomData,
        }
    }

    /// Creates a new tensor from a vector.
    ///
    /// `Mapping` determines the axes of the tensor, and `data` is a flat vector containing the tensor elements in the order sorted by the axes.
    pub fn from_vec<Mapping: M>(data: Vec<Opt<D>>) -> Self {
        // Construct axes from mapping.
        let axes = gen_axes::<Mapping>();

        // Construct data array from vector.
        let shape = axes.iter().map(|term| term.modulo).collect::<Vec<usize>>();
        let data = ArrayD::from_shape_vec(shape, data).expect("Data length does not match tensor shape.");

        Self {
            axes,
            data,
            _marker: PhantomData,
        }
    }

    /// Creates a new tensor from a mapping and a function.
    pub fn from_fn<Mapping: M, F>(mut f: F) -> Self
    where
        F: FnMut(&Vec<Term>, &IxDyn) -> Opt<D>,
    {
        // Construct axes from mapping.
        let axes = gen_axes::<Mapping>();

        // Construct data array from function.
        let shape = axes.iter().map(|term| term.modulo).collect::<Vec<usize>>();
        let data = ArrayD::from_shape_fn(shape, |idx| f(&axes, &idx));

        Self {
            axes,
            data,
            _marker: PhantomData,
        }
    }

    /// Applies a unary function to each element of the tensor.
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = TensorValue::<i32>::from_vec::<m![A, B]>(...);
    /// let doubled = tensor.map(|&x| x * Opt::Init(2));
    /// ```
    pub fn map<D2: Scalar, F>(&self, f: F) -> RawTensor<D2>
    where
        F: FnMut(&Opt<D>) -> Opt<D2>,
    {
        let data = self.data.map(f);
        RawTensor {
            axes: self.axes.clone(),
            data,
            _marker: PhantomData,
        }
    }

    /// Applies a binary function element-wise to two tensors with the same shape.
    ///
    /// # Examples
    /// ```ignore
    /// let a = RawTensor::<i32>::from_vec::<m![A, B]>(...);
    /// let b = RawTensor::<i32>::from_vec::<m![A, B]>(...);
    /// let sum = a.zip_with(&b, |x, y| x + y);
    /// ```
    pub fn zip_with<D2: Scalar, D3: Scalar, F>(&self, other: &RawTensor<D2>, f: F) -> RawTensor<D3>
    where
        F: Fn(Opt<D>, Opt<D2>) -> Opt<D3>,
    {
        assert_eq!(
            self.axes, other.axes,
            "Tensors must have the same axes for element-wise binary operations"
        );

        let data = ndarray::Zip::from(&self.data)
            .and(&other.data)
            .map_collect(|&a, &b| f(a, b));

        RawTensor {
            axes: self.axes.clone(),
            data,
            _marker: PhantomData,
        }
    }

    /// Reduces axes not in `retain_axes` using a custom binary function and identity.
    pub fn reduce(
        &self,
        retain_axes: &[Term],
        reduce_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>,
        identity: Opt<D>,
    ) -> RawTensor<D> {
        let reduce_axes: Vec<Term> = self
            .axes
            .iter()
            .filter(|src_term| !retain_axes.contains(src_term))
            .cloned()
            .collect();
        self.reduce_for(&reduce_axes, reduce_fn, identity)
    }

    /// Performs reduction (sum) over axes.
    /// The axes to retain are specified by their terms.
    pub fn reduce_add(&self, retain_axes: &[Term]) -> RawTensor<D> {
        self.reduce(retain_axes, |a, b| a + b, Opt::zero())
    }

    /// Performs reduction over specified axes using a custom binary function and identity.
    /// The axes to reduce are specified by their terms.
    fn reduce_for(
        &self,
        reduce_axes: &[Term],
        reduce_fn: impl Fn(Opt<D>, Opt<D>) -> Opt<D>,
        identity: Opt<D>,
    ) -> RawTensor<D> {
        // Find indices of axes to reduce.
        let reduce_indices: Vec<usize> = reduce_axes
            .iter()
            .filter_map(|reduce_term| self.axes.iter().position(|axis_term| axis_term == reduce_term))
            .collect();

        // Compute new axes (excluding reduced axes).
        let axes: Vec<Term> = self
            .axes
            .iter()
            .enumerate()
            .filter_map(|(idx, term)| {
                if reduce_indices.contains(&idx) {
                    None
                } else {
                    Some(term.clone())
                }
            })
            .collect();

        // Perform reduction by summing over the specified axes.
        let data = if reduce_indices.is_empty() {
            // No reduction needed
            self.data.clone()
        } else if reduce_indices.len() == self.axes.len() {
            // Reducing all axes to a scalar
            let sum_value = self.data.sum();
            ArrayD::from_shape_vec(IxDyn(&[]), vec![sum_value]).expect("Failed to create scalar array")
        } else {
            // Partial reduction
            // Sort and deduplicate indices, then reduce in reverse order
            let mut sorted_indices = reduce_indices.clone();
            sorted_indices.sort_unstable();
            sorted_indices.dedup();

            // Validate indices
            for &idx in &sorted_indices {
                if idx >= self.axes.len() {
                    panic!(
                        "[TensorValue::reduce] Invalid axis index: {} (tensor has {} axes)\n\
                         axes: {:?}\n\
                         reduce_axes: {:?}\n\
                         reduce_indices: {:?}",
                        idx,
                        self.axes.len(),
                        self.axes,
                        reduce_axes,
                        sorted_indices
                    );
                }
            }

            let mut data = self.data.clone();
            for &axis_idx in sorted_indices.iter().rev() {
                if axis_idx >= data.ndim() {
                    panic!(
                        "[TensorValue::reduce] axis_idx {} >= data.ndim() {}\n\
                         current data shape: {:?}\n\
                         axes: {:?}\n\
                         reduce_indices: {:?}",
                        axis_idx,
                        data.ndim(),
                        data.shape(),
                        self.axes,
                        sorted_indices
                    );
                }
                data = data.fold_axis(Axis(axis_idx), identity, |&acc, &val| reduce_fn(acc, val));
            }
            data
        };

        Self {
            axes,
            data,
            _marker: PhantomData,
        }
    }

    /// Broadcasts and writes data from another tensor based on the given mappings and offsets.
    ///
    /// # Arguments
    /// * `src` - Source tensor to read data from.
    /// * `unicast` - Mapping for 1-to-1 mapping dimensions.
    /// * `broadcast` - Mapping for broadcasted dimensions.
    /// * `src_offset` - Offset to apply to the source tensor indices.
    /// * `dst_offset` - Offset to apply to the destination (self) tensor indices.
    pub fn write_broadcast(
        &mut self,
        src: &Self,
        unicast: FMapping,
        broadcast: FMapping,
        src_offset: &Index,
        dst_offset: &Index,
    ) {
        // Iterate over all unicast indexes.
        for index in Index::new().gen_indexes(unicast) {
            // Read value from source tensor at the index with offset.
            let mut src_index = index.clone();
            src_index.add(src_offset.clone());
            let value = src.read_index(src_index);

            // Calculate the base destination index with offset.
            let mut dst_index = index;
            dst_index.add(dst_offset.clone());

            // Write value to all broadcasted indexes in the destination tensor.
            for dst_index in dst_index.gen_indexes(broadcast.clone()) {
                self.write_index(dst_index, value);
            }
        }
    }

    /// Scatters elements from `src` into `self` at positions given by `indices`.
    ///
    /// ```text
    /// src:    [N, K, V]
    /// dst:    [N, X, V]
    /// key:    K
    /// idices: [N, K]
    ///
    /// for n, k:
    ///     dst[n][indices[n,k]][v] = src[n][k][v]
    /// ```
    ///
    /// Decomposition via divide:
    /// 1. `src / key`     = payload [N, V] — axes preserved across scatter
    /// 2. `dst / payload` = target  [X]    — scatter target axis
    pub(crate) fn write_scatter(
        &mut self,
        src: &Self,
        src_mapping: FMapping,
        dst_mapping: FMapping,
        key: FMapping,
        indices: &[usize],
    ) {
        let payload = src_mapping
            .clone()
            .divide_relaxed(key.clone())
            .exact()
            .unwrap_or_else(|e| panic!("Scatter key `{key:?}` not found in source `{src_mapping:?}`: {e:?}"))
            .dividend_residue;
        let dst_residue = dst_mapping
            .clone()
            .divide_relaxed(payload.clone())
            .exact()
            .unwrap_or_else(|e| panic!("Destination `{dst_mapping:?}` missing payload axes `{payload:?}`: {e:?}"))
            .dividend_residue;
        let (dst_term, _) = dst_residue
            .into_inner()
            .into_iter()
            .find_map(|f| match f {
                Factor::Term { inner, resize } => Some((inner, resize)),
                Factor::Padding { .. } => None,
            })
            .expect("Destination has no scatter target axis after removing payload");

        for payload_index in Index::new().gen_indexes(payload) {
            for (key_pos, key_index) in Index::new().gen_indexes(key.clone()).into_iter().enumerate() {
                let mut src_index = payload_index.clone();
                src_index.add(key_index);
                let value = src.read_index(src_index);

                let mut dst_index = payload_index.clone();
                dst_index.add_term(dst_term.clone(), indices[key_pos]);
                self.write_index(dst_index, value);
            }
        }
    }

    /// Reads the tensor value at the given index.
    pub(crate) fn read_index(&self, index: Index) -> Opt<D> {
        // Finalize the index before reading.
        let RResult::ROk(index) = index.finalize() else {
            return Opt::Uninit;
        };
        assert!(
            self.axes.iter().zip(index.iter()).all(|(a, (b, _))| a == b),
            "Index terms ({:?}) do not match tensor axes ({:?}).",
            index,
            self.axes
        );

        // Read the value from the data array.
        *self
            .data
            .get(index.into_iter().map(|(_, v)| v).collect::<Vec<usize>>().as_slice())
            .expect("Index out of bounds.")
    }

    /// Writes the tensor value at the given index.
    pub(crate) fn write_index(&mut self, index: Index, value: Opt<D>) {
        // Finalize the index before reading.
        let RResult::ROk(index) = index.finalize() else {
            return;
        };
        assert!(
            self.axes.iter().zip(index.iter()).all(|(a, (b, _))| a == b),
            "Index terms do not match tensor axes."
        );

        // Write the value to the data array.
        *self
            .data
            .get_mut(index.into_iter().map(|(_, v)| v).collect::<Vec<usize>>().as_slice())
            .expect("Index out of bounds.") = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_tensor_zip_with() {
        axes![A = 2, B = 3];

        let t1 = RawTensor::<i32>::from_vec::<m![A, B]>((1..7).map(Opt::Init).collect::<Vec<_>>());
        let t2 = RawTensor::<i32>::from_vec::<m![A, B]>((2..8).map(Opt::Init).collect::<Vec<_>>());
        let result = t1.zip_with(&t2, |a, b| a * b);
        let expected = RawTensor::<i32>::from_vec::<m![A, B]>(
            [2, 6, 12, 20, 30, 42].into_iter().map(Opt::Init).collect::<Vec<_>>(),
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_reduce() {
        axes![A = 2, B = 3];

        let t = RawTensor::<i32>::from_vec::<m![A, B]>((1..7).map(Opt::Init).collect::<Vec<_>>());

        // Reduce over B axis (retain A only)
        let retain_axes = gen_axes::<m![A]>();
        let result = t.reduce_add(&retain_axes);

        let expected = RawTensor::<i32>::from_vec::<m![A]>([6, 15].into_iter().map(Opt::Init).collect::<Vec<_>>());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_reduce_multiple_axes() {
        axes![A = 2, B = 2, C = 2];

        let t = RawTensor::<i32>::from_vec::<m![A, B, C]>((1..9).map(Opt::Init).collect::<Vec<_>>());

        // Reduce over B and C axes (retain A only)
        let retain_axes = gen_axes::<m![A]>();
        let result = t.reduce_add(&retain_axes);

        let expected = RawTensor::<i32>::from_vec::<m![A]>([10, 26].into_iter().map(Opt::Init).collect::<Vec<_>>());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_map_elements() {
        axes![A = 2, B = 3];

        let t = RawTensor::<i32>::from_vec::<m![A, B]>((1..7).map(Opt::Init).collect::<Vec<_>>());
        let result = t.map(|&x| x * Opt::Init(2));

        let expected =
            RawTensor::<i32>::from_vec::<m![A, B]>([2, 4, 6, 8, 10, 12].into_iter().map(Opt::Init).collect::<Vec<_>>());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_zip_with_custom_operation() {
        axes![A = 2, B = 3];

        let t1 = RawTensor::<i32>::from_vec::<m![A, B]>((1..7).map(Opt::Init).collect::<Vec<_>>());
        let t2 = RawTensor::<i32>::from_vec::<m![A, B]>((2..8).map(Opt::Init).collect::<Vec<_>>());

        // Custom operation: (a * b) + 1
        let result = t1.zip_with(&t2, |a, b| match (a, b) {
            (Opt::Init(x), Opt::Init(y)) => Opt::Init(x * y + 1),
            _ => Opt::Uninit,
        });

        let expected = RawTensor::<i32>::from_vec::<m![A, B]>(
            [3, 7, 13, 21, 31, 43].into_iter().map(Opt::Init).collect::<Vec<_>>(),
        );
        assert_eq!(result, expected);
    }
}
