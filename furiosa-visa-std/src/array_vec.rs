use std::ops::{Deref, DerefMut};

use arrayvec::ArrayVec as ArrayVecInner;

/// Compile-time check that N <= CAP.
const fn check_capacity<const N: usize, const CAP: usize>() {
    assert!(N <= CAP, "Array size exceeds capacity");
}

/// Wrapper around `arrayvec::ArrayVec` that allows initialization with arrays smaller than capacity.
///
/// # Example
/// ```ignore
/// let arr: ArrayVec<i32, 8> = ArrayVec::new([1, 2, 3]); // capacity 8, length 3
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayVec<T, const CAP: usize>(ArrayVecInner<T, CAP>);

impl<T, const CAP: usize> Default for ArrayVec<T, CAP> {
    fn default() -> Self {
        Self(ArrayVecInner::new())
    }
}

impl<T, const CAP: usize> ArrayVec<T, CAP> {
    /// Creates a new ArrayVec from an array. The array size N can be smaller than capacity CAP.
    /// Compile-time error if N > CAP.
    pub fn new<const N: usize>(arr: [T; N]) -> Self {
        check_capacity::<N, CAP>();
        let mut inner = ArrayVecInner::new();
        for item in arr {
            inner.push(item);
        }
        Self(inner)
    }

    /// Creates an empty ArrayVec.
    pub fn empty() -> Self {
        Self(ArrayVecInner::new())
    }
}

impl<T, const CAP: usize> Deref for ArrayVec<T, CAP> {
    type Target = ArrayVecInner<T, CAP>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const CAP: usize> DerefMut for ArrayVec<T, CAP> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const CAP: usize> IntoIterator for ArrayVec<T, CAP> {
    type Item = T;
    type IntoIter = arrayvec::IntoIter<T, CAP>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, const CAP: usize> IntoIterator for &'a ArrayVec<T, CAP> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T, const CAP: usize> IntoIterator for &'a mut ArrayVec<T, CAP> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}
