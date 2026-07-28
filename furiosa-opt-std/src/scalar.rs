//! Scalar types.

use furiosa_opt_macro::primitive;
use num_traits::{Num, One, Zero};
use rand::distr::StandardUniform;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Rem, Sub};

/// A trait for scalar types.
pub trait Scalar: ndarray::LinalgScalar + Debug + Clone + Copy + PartialEq + Num + Send + Sync {
    /// Number of bits per element.
    const BITS: usize;

    /// Returns the byte size for `length` elements of this scalar type.
    ///
    /// Panics if the total bit count is not a multiple of 8.
    fn size_in_bytes_from_length(length: usize) -> usize {
        crate::constraints::size_in_bytes(Self::BITS, length)
    }

    /// The buffer byte length for `n` elements: the size of [`to_buf`](Self::to_buf)'s output, the
    /// run [`load`](Self::load) / [`store`](Self::store) address per element. Byte-centric by default
    /// (`n * size_of`); the sub-byte packers ([`i4`]/[`f4e2m1`]) override it to their `BITS`-packed
    /// size. Wider than [`size_in_bytes_from_length`](Self::size_in_bytes_from_length) for a staging
    /// type ([`i5`]/[`i9`]), whose `i8`/`i16` host form exceeds its narrower `BITS` wire width.
    fn buf_bytes(n: usize) -> usize {
        n * std::mem::size_of::<Self>()
    }

    /// Reads element `i` from a packed byte image. This and [`store`](Self::store) are the sole place the
    /// byte-multiple-vs-sub-byte distinction lives: the default is byte-centric (element `i` owns its own
    /// `Self::BITS / 8`-byte run of the little-endian image), and only [`f4e2m1`] overrides it to unpack a
    /// nibble. Every consumer above (the packed [`BufStorage`](crate::storage::BufStorage) buffer,
    /// [`to_buf`](Self::to_buf)) drives elements through this pair and never branches on
    /// width itself.
    ///
    /// The default is the whole in-memory value: for a byte-multiple width this is a direct copy of the
    /// little-endian bytes, byte-identical to the raw `&[Self]` image (low element = low byte).
    fn load(bytes: &[u8], i: usize) -> Self {
        let n = std::mem::size_of::<Self>();
        // SAFETY: `Self: Copy` with no padding-sensitive byte-image invariant, the same little-endian view
        // `to_buf` / `Buffer::from_slice` take of a `&[Self]`. `bytes[i*n..(i+1)*n]` is a full
        // `Self`-sized run, read unaligned into a `Self`.
        unsafe { std::ptr::read_unaligned(bytes[i * n..(i + 1) * n].as_ptr() as *const Self) }
    }

    /// Writes element `i` into a packed byte image, the inverse of [`load`](Self::load) and the other half
    /// of the sole width-branching site (see [`load`](Self::load)). The default lays the whole
    /// little-endian value into element `i`'s `Self::BITS / 8`-byte run; only [`f4e2m1`] overrides it to
    /// pack a nibble.
    fn store(bytes: &mut [u8], i: usize, v: Self) {
        let n = std::mem::size_of::<Self>();
        // SAFETY: `Self: Copy` sized `n`; `&v` is a valid `Self`-sized little-endian image, copied into
        // element `i`'s run (the exact inverse of `load`'s read).
        let raw = unsafe { std::slice::from_raw_parts(&v as *const Self as *const u8, n) };
        bytes[i * n..(i + 1) * n].copy_from_slice(raw);
    }

    /// Returns the number of elements that fit in `bytes` bytes.
    ///
    /// Panics if the byte count does not evenly divide into whole elements.
    fn length_from_bytes(bytes: usize) -> usize {
        assert_eq!(
            (bytes * 8) % Self::BITS,
            0,
            "bytes must correspond to a whole number of elements"
        );
        (bytes * 8) / Self::BITS
    }

    /// Serializes `vals` into the dense physical DRAM byte image: the exact bytes that would sit
    /// in HBM / DM for this element stream, sized [`size_in_bytes_from_length`](Self::size_in_bytes_from_length).
    ///
    /// This is the single reconciliation point between the type's *nominal* in-memory
    /// representation (a sub-byte scalar such as [`f4e2m1`] / [`i4`] still occupies a whole byte in
    /// a `&[Self]` slice) and its *physical* on-device layout, where the codes are dense-packed by
    /// bit width (`Precision::FourBits` for the 4-bit scalars). For a byte-multiple width the packed
    /// image IS the raw little-endian `&[Self]` bytes, so this default is one `memcpy` of the whole
    /// slice (the whole-slice form of [`store`](Self::store)); a sub-byte width ([`f4e2m1`] / [`i4`])
    /// overrides it to emit two codes per byte (low element in the low nibble, half the length), the
    /// one place the even-count check lives (a byte-multiple width needs none). The bit-width stays
    /// confined to these two arms, so every downstream consumer stays generic over `D: Scalar` and
    /// reads the size this returns (the LIR executor's `ElementType::size_in_bytes_with_length`
    /// computes the same `n * BITS` bit count), keeping the stream packed all the way to the fetch
    /// adapter, the sole stage that unpacks it.
    fn to_buf(vals: &[Self]) -> Vec<u8> {
        // Byte-multiple: the packed image is the raw little-endian `&[Self]` bytes (the identical view
        // `load` / `store` take of one element), so serialize the whole slice in a single memcpy rather
        // than element-by-element. The sub-byte override below handles `f4e2m1` / `i4`.
        //
        // SAFETY: `Self: Copy` with no padding-sensitive byte-image invariant; `&[Self]` reinterpreted as
        // bytes is the same little-endian image `store` lays down per element, over `size_of_val(vals)` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, std::mem::size_of_val(vals)) };
        bytes.to_vec()
    }
}

/// A byte-aligned [`Scalar`] that can be decoded from its little-endian byte representation.
///
/// Excludes sub-byte scalars like [`i4`] for which a single element cannot be addressed at a
/// byte boundary. This is what [`crate::tensor::memory::HostTensor::from_safetensors`] requires,
/// matching the set of dtypes safetensors itself can carry.
pub trait ScalarBytes: Scalar {
    /// Decodes one element from `bytes`; `bytes.len()` must equal `Self::BITS / 8`.
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

/// A [`Scalar`] that may live in host/HBM/DM memory or be materialized by the
/// commit, to-TRF, or transpose paths.
///
/// Every real tensor dtype implements this. The zero-point-subtracted
/// contraction-engine stagings [`i5`]/[`i9`] deliberately do **not**: as a
/// stream they exist only between `fetch_zero_point_sub` and `contract_outer`,
/// so the host/HBM/DM carriers and the commit / to-TRF / transpose entry points
/// bound `MaterializableScalar` to make storing or re-routing an i5/i9 stream a
/// compile error. The to-VRF / Vector / Cast paths exclude them separately via
/// `VeScalar`. (An i5/i9 may still be a contraction weight resident in the TRF.)
///
/// A real dtype satisfies the bound:
/// ```
/// use furiosa_opt_std::prelude::*;
/// fn materialize<D: MaterializableScalar>() {}
/// materialize::<i8>();
/// ```
/// A staging type does not, so materializing it fails to compile:
/// ```compile_fail
/// use furiosa_opt_std::prelude::*;
/// fn materialize<D: MaterializableScalar>() {}
/// materialize::<i9>();
/// ```
pub trait MaterializableScalar: Scalar {}

/// Implements [`Scalar`] (`BITS = size_of * 8`) and [`ScalarBytes`] (delegating to the type's inherent
/// `from_le_bytes([u8; BITS/8])`) for each byte-aligned type. `bf16`/`f8e4m3`/`f8e5m2` are defined later
/// in the module and supply their own inherent `from_le_bytes`. Sub-byte `i4` is hand-written below.
macro_rules! impl_scalar {
    ($($t:ty);* $(;)?) => {
        $(
            impl Scalar for $t {
                const BITS: usize = ::std::mem::size_of::<$t>() * 8;
            }
            impl ScalarBytes for $t {
                fn from_le_bytes(bytes: &[u8]) -> Self {
                    // Convert the slice to the inherent `from_le_bytes`'s fixed-length array, which both
                    // checks the length and disambiguates from this trait method (the delegate below
                    // is the array-taking inherent, not a recursion).
                    let arr = bytes.try_into().unwrap_or_else(|_| {
                        panic!(
                            "{} expects {} bytes, got {}",
                            stringify!($t),
                            <$t as Scalar>::BITS / 8,
                            bytes.len()
                        )
                    });
                    <$t>::from_le_bytes(arr)
                }
            }
            impl MaterializableScalar for $t {}
        )*
    };
}

// The byte-aligned scalars.
impl_scalar! {
    i8;
    u8;
    i16;
    i32;
    f8e4m3;
    f8e5m2;
    bf16;
    f32;
}

/// A data type that can be either initialized or uninitialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opt<D> {
    /// Initialized value.
    Init(D),
    /// Uninitialized value.
    Uninit,
}

impl<D> Opt<D> {
    /// Maps the initialized value using the provided function, or returns uninitialized.
    #[inline]
    pub fn map<D2>(self, f: impl FnOnce(D) -> D2) -> Opt<D2> {
        match self {
            Opt::Init(val) => Opt::Init(f(val)),
            Opt::Uninit => Opt::Uninit,
        }
    }

    /// Combines two Opt values with a function. Returns Init only if both are Init.
    #[inline]
    pub fn zip_map<D2, R>(self, other: Opt<D2>, f: impl FnOnce(D, D2) -> R) -> Opt<R> {
        match (self, other) {
            (Opt::Init(a), Opt::Init(b)) => Opt::Init(f(a, b)),
            _ => Opt::Uninit,
        }
    }

    /// Returns the initialized value, or panics if uninitialized.
    pub fn unwrap(self) -> D {
        let Opt::Init(val) = self else {
            panic!("Called unwrap on an uninitialized Opt value.");
        };
        val
    }
}

// `Opt` carries `Add` / `Mul` only for the `num_traits::Zero` / `One` supertraits the fold seed needs
// (`Opt::Init(zero())`); both compose via `zip_map`.
impl<D: Add<Output = D>> Add for Opt<D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Opt<D> {
        self.zip_map(rhs, |l, r| l + r)
    }
}

impl<D: Mul<Output = D>> Mul for Opt<D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Opt<D> {
        self.zip_map(rhs, |l, r| l * r)
    }
}

impl<D: Zero> Zero for Opt<D> {
    fn zero() -> Self {
        Opt::Init(D::zero())
    }

    fn is_zero(&self) -> bool {
        let Opt::Init(val) = self else {
            return false;
        };
        val.is_zero()
    }
}

impl<D: One> One for Opt<D> {
    fn one() -> Self {
        Opt::Init(D::one())
    }
}

/// 16-bit brain floating point type.
#[primitive(bf16)]
#[expect(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct bf16(half::bf16);

impl Zero for bf16 {
    fn zero() -> Self {
        bf16(half::bf16::from_f32(0.0))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for bf16 {
    fn one() -> Self {
        bf16(half::bf16::from_f32(1.0))
    }
}

impl Add<Self> for bf16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        bf16(self.0 + rhs.0)
    }
}

impl Sub<Self> for bf16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        bf16(self.0 - rhs.0)
    }
}

impl Mul<Self> for bf16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        bf16(self.0 * rhs.0)
    }
}

impl Div<Self> for bf16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        bf16(self.0 / rhs.0)
    }
}

impl Rem<Self> for bf16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        bf16(self.0 % rhs.0)
    }
}

impl Num for bf16 {
    type FromStrRadixErr = <half::bf16 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(bf16(half::bf16::from_str_radix(str, radix)?))
    }
}

impl rand::distr::Distribution<bf16> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let val: f32 = rng.random_range(-1.0..1.0);
        bf16(half::bf16::from_f32(val))
    }
}

impl bf16 {
    /// Decodes one element from its little-endian bytes; the inherent peer the `impl_scalar!`-generated
    /// [`ScalarBytes`] delegates to (matching `std`'s `from_le_bytes([u8; N])` shape).
    pub(crate) fn from_le_bytes(bytes: [u8; 2]) -> Self {
        bf16(half::bf16::from_bits(u16::from_le_bytes(bytes)))
    }

    /// Creates `bf16` from `f32` in a `const` context (for compile-time fill values). Byte-identical
    /// to the runtime conversion: `half`'s own `from_f32` forwards to the `from_f32_const` used here.
    pub const fn from_f32(val: f32) -> Self {
        bf16(half::bf16::from_f32_const(val))
    }

    /// Converts to `f32`.
    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }
}

impl From<bf16> for f32 {
    fn from(val: bf16) -> Self {
        val.to_f32()
    }
}

/// 8-bit floating point type with 4-bit exponent (E4M3).
#[primitive(f8e4m3)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f8e4m3(u8);

impl Zero for f8e4m3 {
    fn zero() -> Self {
        f8e4m3(crate::float::F8E4_ZERO)
    }

    fn is_zero(&self) -> bool {
        crate::float::f8_e4_is_zero(self.0)
    }
}

impl One for f8e4m3 {
    fn one() -> Self {
        f8e4m3(crate::float::F8E4_ONE)
    }
}

impl Add<Self> for f8e4m3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub<Self> for f8e4m3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul<Self> for f8e4m3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div<Self> for f8e4m3 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Rem<Self> for f8e4m3 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl Num for f8e4m3 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_f32(f32::from_str_radix(str, radix)?))
    }
}

impl rand::distr::Distribution<f8e4m3> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f8e4m3 {
        let val: f32 = rng.random_range(-1.0..1.0);
        f8e4m3::from_f32(val)
    }
}

impl f8e4m3 {
    /// Decodes one element from its little-endian byte; the inherent peer the `impl_scalar!`-generated
    /// [`ScalarBytes`] delegates to (matching `std`'s `from_le_bytes([u8; N])` shape).
    pub(crate) fn from_le_bytes(bytes: [u8; 1]) -> Self {
        f8e4m3(bytes[0])
    }

    /// Creates `f8e4m3` from `f32` in a `const` context. Byte-identical to the runtime conversion
    /// (the E4M3 rounding is pure integer arithmetic, so the same code is `const`-evaluable).
    pub const fn from_f32(val: f32) -> Self {
        f8e4m3(crate::float::f8_e4_from_f32(val))
    }

    /// Converts to `f32`.
    pub fn to_f32(self) -> f32 {
        crate::float::f8_e4_to_f32(self.0)
    }
}

impl From<f8e4m3> for f32 {
    fn from(val: f8e4m3) -> Self {
        val.to_f32()
    }
}

/// 8-bit floating point type with 5-bit exponent (E5M2).
#[primitive(f8e5m2)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f8e5m2(u8);

impl Zero for f8e5m2 {
    fn zero() -> Self {
        f8e5m2(crate::float::F8E5_ZERO)
    }

    fn is_zero(&self) -> bool {
        crate::float::f8_e5_is_zero(self.0)
    }
}

impl One for f8e5m2 {
    fn one() -> Self {
        f8e5m2(crate::float::F8E5_ONE)
    }
}

impl Add<Self> for f8e5m2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub<Self> for f8e5m2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul<Self> for f8e5m2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div<Self> for f8e5m2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Rem<Self> for f8e5m2 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl Num for f8e5m2 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_f32(f32::from_str_radix(str, radix)?))
    }
}

impl rand::distr::Distribution<f8e5m2> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f8e5m2 {
        let val: f32 = rng.random_range(-1.0..1.0);
        f8e5m2::from_f32(val)
    }
}

impl f8e5m2 {
    /// Decodes one element from its little-endian byte; the inherent peer the `impl_scalar!`-generated
    /// [`ScalarBytes`] delegates to (matching `std`'s `from_le_bytes([u8; N])` shape).
    pub(crate) fn from_le_bytes(bytes: [u8; 1]) -> Self {
        f8e5m2(bytes[0])
    }

    /// Creates `f8e5m2` from `f32`.
    pub fn from_f32(val: f32) -> Self {
        f8e5m2(crate::float::f8_e5_from_f32(val))
    }

    /// Converts to `f32`.
    pub fn to_f32(self) -> f32 {
        crate::float::f8_e5_to_f32(self.0)
    }
}

impl From<f8e5m2> for f32 {
    fn from(val: f8e5m2) -> Self {
        val.to_f32()
    }
}

/// 4-bit signed integer type.
///
/// Stored as `i8` with sign-extension: valid range is `[-8, 7]`.
#[primitive(i4)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct i4(i8);

impl Zero for i4 {
    fn zero() -> Self {
        i4(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for i4 {
    fn one() -> Self {
        i4(1)
    }
}

impl Add<Self> for i4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 + rhs.0)
    }
}

impl Sub<Self> for i4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 - rhs.0)
    }
}

impl Mul<Self> for i4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 * rhs.0)
    }
}

impl Div<Self> for i4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 / rhs.0)
    }
}

impl Rem<Self> for i4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 % rhs.0)
    }
}

impl Num for i4 {
    type FromStrRadixErr = <i8 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_lsb(i8::from_str_radix(str, radix)?))
    }
}

// `BITS = 4` is explicit (`size_of` would round the `i8` storage up to 8); `i4` is not a `ScalarBytes`
// since a single 4-bit element is not byte-addressable.
impl Scalar for i4 {
    const BITS: usize = 4;

    /// Two nibbles per byte, low element in the low nibble; the packed nibble is a signed 4-bit code,
    /// sign-extended back into `i4` on load. `from_i32` routes through `from_lsb`, which sign-extends
    /// the low 4 bits into the `[-8, 7]` range.
    fn load(bytes: &[u8], i: usize) -> Self {
        i4::from_i32(get_nibble(bytes, i) as i32)
    }

    /// Packs element `i`'s low nibble into its half of byte `i / 2`, leaving the sibling nibble untouched.
    fn store(bytes: &mut [u8], i: usize, v: i4) {
        set_nibble(bytes, i, v.to_i32() as u8);
    }

    /// Two codes per byte (see [`store`](Self::store)); the sub-byte override of the memcpy default.
    fn to_buf(vals: &[Self]) -> Vec<u8> {
        pack_nibbles(vals)
    }

    fn buf_bytes(n: usize) -> usize {
        Self::size_in_bytes_from_length(n)
    }
}

impl i4 {
    const fn from_lsb(n: i8) -> Self {
        i4((n << 4) >> 4)
    }

    /// Creates `i4` from `i32`, at compile time (sign-truncation to the low 4 bits).
    pub const fn from_i32(val: i32) -> Self {
        Self::from_lsb(val as i8)
    }

    /// Converts to `i32`.
    pub fn to_i32(self) -> i32 {
        i32::from(self.0)
    }
}

impl From<i4> for i32 {
    fn from(val: i4) -> Self {
        val.to_i32()
    }
}

/// 5-bit signed integer: the zero-point-subtracted staging of [`i4`].
///
/// Produced by `fetch_zero_point_sub`, consumed only by `contract_outer` (it is
/// not [`MaterializableScalar`], so it cannot be committed or routed to another
/// engine). `i4 - i4` needs the extra bit: `[-8,7] - [-8,7] = [-15,15]` fits in
/// `i5`'s `[-16,15]`. Stored as `i8` with sign-extension.
#[primitive(i5)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct i5(i8);

impl Zero for i5 {
    fn zero() -> Self {
        i5(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for i5 {
    fn one() -> Self {
        i5(1)
    }
}

// Arithmetic computes in i32 and narrows back, because the 5-bit result stored
// in `i8` would overflow mid-op (`(-16) * (-16) = 256` exceeds `i8::MAX`).
impl Add<Self> for i5 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() + rhs.to_i32())
    }
}

impl Sub<Self> for i5 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() - rhs.to_i32())
    }
}

impl Mul<Self> for i5 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() * rhs.to_i32())
    }
}

impl Div<Self> for i5 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() / rhs.to_i32())
    }
}

impl Rem<Self> for i5 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() % rhs.to_i32())
    }
}

impl Num for i5 {
    type FromStrRadixErr = <i8 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_lsb(i8::from_str_radix(str, radix)?))
    }
}

impl Scalar for i5 {
    // Packs as i4 in storage; the extra bit lives in the contraction engine, not memory.
    const BITS: usize = 4;
}

impl i5 {
    // Sign-extend the low 5 bits of an i8 (8 - 5 = 3).
    fn from_lsb(n: i8) -> Self {
        i5((n << 3) >> 3)
    }

    /// Creates `i5` from `i32`.
    pub fn from_i32(val: i32) -> Self {
        Self::from_lsb(val as i8)
    }

    /// Converts to `i32`.
    pub fn to_i32(self) -> i32 {
        i32::from(self.0)
    }
}

impl From<i5> for i32 {
    fn from(val: i5) -> Self {
        val.to_i32()
    }
}

/// 9-bit signed integer: the zero-point-subtracted staging of [`i8`].
///
/// Produced by `fetch_zero_point_sub`, consumed only by `contract_outer` (it is
/// not [`MaterializableScalar`], so it cannot be committed or routed to another
/// engine). `i8 - i8` needs the extra bit: `[-128,127] - [-128,127] =
/// [-255,255]` fits in `i9`'s `[-256,255]`. Stored as `i16` with sign-extension.
#[primitive(i9)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct i9(i16);

impl Zero for i9 {
    fn zero() -> Self {
        i9(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for i9 {
    fn one() -> Self {
        i9(1)
    }
}

// Arithmetic computes in i32 and narrows back, because the 9-bit result stored
// in `i16` would overflow mid-op (`255 * 255 = 65025` exceeds `i16::MAX`).
impl Add<Self> for i9 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() + rhs.to_i32())
    }
}

impl Sub<Self> for i9 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() - rhs.to_i32())
    }
}

impl Mul<Self> for i9 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() * rhs.to_i32())
    }
}

impl Div<Self> for i9 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() / rhs.to_i32())
    }
}

impl Rem<Self> for i9 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_i32(self.to_i32() % rhs.to_i32())
    }
}

impl Num for i9 {
    type FromStrRadixErr = <i16 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_lsb(i16::from_str_radix(str, radix)?))
    }
}

impl Scalar for i9 {
    // Packs as i8 in storage; the extra bit lives in the contraction engine, not memory.
    const BITS: usize = 8;
}

impl i9 {
    // Sign-extend the low 9 bits of an i16 (16 - 9 = 7).
    fn from_lsb(n: i16) -> Self {
        i9((n << 7) >> 7)
    }

    /// Creates `i9` from `i32`.
    pub fn from_i32(val: i32) -> Self {
        Self::from_lsb(val as i16)
    }

    /// Converts to `i32`.
    pub fn to_i32(self) -> i32 {
        i32::from(self.0)
    }
}

impl From<i9> for i32 {
    fn from(val: i9) -> Self {
        val.to_i32()
    }
}

// `i4` is hand-written (not driven by `impl_scalar!`), so it needs the marker here;
// the macro grants it to the byte-aligned dtypes, and the i5/i9 stagings intentionally lack it.
impl MaterializableScalar for i4 {}

/// 4-bit float in `e2m1` encoding (NVFP4 / MXFP4 element type).
///
/// One nibble in the low 4 bits of a `u8`; the upper 4 bits are ignored. The value
/// set is `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6, -0}`, the hardware e2m1 value set.
///
/// The hardware decodes `f4e2m1` only through the Fetch Adapter's
/// table-lookup stage (`fetch_table_lookup`): the nibble
/// indexes a 16-entry table to [`f8e4m3`] (paired-key table, 4b->8b only; wider
/// floats and the per-block scale come downstream). See the book chapter
/// `computing-tensors/fetch-adapter.md` (the "Table Lookup" section).
#[primitive(f4e2m1)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
pub struct f4e2m1(u8);

/// e2m1 nibble -> `f8e4m3` bit pattern. A hand-maintained mirror of the hardware
/// F4E2 -> F8E4 conversion table, kept in sync by hand since this crate cannot depend
/// on the table's source; the `f4e2m1 -> f8e4m3` decode test pins every entry against
/// the independent e2m1 value spec, so drift from either source is caught.
const F4E2M1_TO_F8E4: [u8; 16] = [
    0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c, 0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc,
];

impl f4e2m1 {
    /// The one canonicalizing constructor: masks to the low nibble (the upper 4
    /// bits are discarded) so the `self.0 < 16` invariant holds. Every other
    /// constructor routes here, keeping `to_f8e4m3`'s array index in range
    /// (mirrors `i4::from_lsb`).
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        f4e2m1(bits & 0xf)
    }

    /// The stored nibble (always `< 16`).
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Decodes the nibble to `f8e4m3` via the 16-entry hardware table. Functional
    /// model of the Fetch Adapter's table-lookup stage for e2m1 keys.
    pub fn to_f8e4m3(self) -> f8e4m3 {
        f8e4m3(F4E2M1_TO_F8E4[self.0 as usize])
    }

    /// Decodes to `f32` (through the `f8e4m3` table). `0x0` yields `0.0`, `0x8` yields `-0.0`.
    pub fn to_f32(self) -> f32 {
        self.to_f8e4m3().to_f32()
    }

    /// Nearest-code encode over the 16-entry value set, ties to the lower code.
    /// Only exists to satisfy the `Num`/`Zero`/`One` bounds; cold path.
    pub fn from_f32(v: f32) -> Self {
        let mut best = 0u8;
        let mut best_err = f32::INFINITY;
        for code in 0..16u8 {
            let err = (Self::from_bits(code).to_f32() - v).abs();
            if err < best_err {
                best_err = err;
                best = code;
            }
        }
        Self::from_bits(best)
    }
}

impl From<f4e2m1> for f32 {
    fn from(val: f4e2m1) -> Self {
        val.to_f32()
    }
}

/// Compares by decoded value, not raw nibble, so `0x0` and `0x8` (both zero) are
/// equal, required for the `Zero` law `a.is_zero() == (a == zero())`.
impl PartialEq for f4e2m1 {
    fn eq(&self, other: &Self) -> bool {
        self.to_f32() == other.to_f32()
    }
}

impl Zero for f4e2m1 {
    fn zero() -> Self {
        Self::from_bits(0x0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0x0 || self.0 == 0x8
    }
}

impl One for f4e2m1 {
    fn one() -> Self {
        Self::from_bits(0x2)
    }
}

// Arithmetic on f4e2m1 exists only to satisfy the Num/Scalar bound; the HW never
// does e2m1 math. Cold: round-trips through f32 and re-encodes via from_f32.
impl Add<Self> for f4e2m1 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub<Self> for f4e2m1 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul<Self> for f4e2m1 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div<Self> for f4e2m1 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Rem<Self> for f4e2m1 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl Num for f4e2m1 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_f32(f32::from_str_radix(str, radix)?))
    }
}

impl rand::distr::Distribution<f4e2m1> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f4e2m1 {
        f4e2m1::from_bits(rng.random_range(0..16u8))
    }
}

impl Scalar for f4e2m1 {
    const BITS: usize = 4;

    /// Two nibbles per byte, low element in the low nibble (the dense on-device 4-bit layout).
    fn load(bytes: &[u8], i: usize) -> Self {
        f4e2m1::from_bits(get_nibble(bytes, i))
    }

    /// Packs element `i`'s nibble into its half of byte `i / 2`, leaving the sibling nibble untouched.
    fn store(bytes: &mut [u8], i: usize, v: f4e2m1) {
        set_nibble(bytes, i, v.to_bits());
    }

    /// Two codes per byte (see [`store`](Self::store)); the sub-byte override of the memcpy default.
    fn to_buf(vals: &[Self]) -> Vec<u8> {
        pack_nibbles(vals)
    }

    fn buf_bytes(n: usize) -> usize {
        Self::size_in_bytes_from_length(n)
    }
}

// `f4e2m1` is hand-written (not driven by `impl_scalar!`), so it needs the marker here too, same as
// `i4` above -- a real, host-materializable device dtype (its `load`/`store`/`to_buf` pack exactly
// its declared `BITS`, consistently, unlike `i5`/`i9`).
impl MaterializableScalar for f4e2m1 {}

/// Reads element `i`'s nibble from a two-codes-per-byte packed image: the low nibble of byte `i / 2`
/// for an even `i`, the high nibble for an odd one. The shared sub-byte load primitive for [`i4`] and
/// [`f4e2m1`].
#[inline]
fn get_nibble(bytes: &[u8], i: usize) -> u8 {
    if i & 1 == 0 {
        bytes[i / 2] & 0x0f
    } else {
        bytes[i / 2] >> 4
    }
}

/// Packs `nib`'s low 4 bits into element `i`'s half of byte `i / 2`, leaving the sibling nibble
/// untouched (a read-modify-write). The shared sub-byte store primitive for [`i4`] and [`f4e2m1`].
#[inline]
fn set_nibble(bytes: &mut [u8], i: usize, nib: u8) {
    let b = &mut bytes[i / 2];
    let nib = nib & 0x0f;
    *b = if i & 1 == 0 {
        (*b & 0xf0) | nib
    } else {
        (*b & 0x0f) | (nib << 4)
    };
}

/// The sub-byte `to_buf`, shared by [`i4`] and [`f4e2m1`]: packs a 4-bit `&[D]` two codes per byte via
/// [`Scalar::store`]. The even-count assert lives here (the sole sub-byte serializer): an odd count has no
/// whole-byte layout.
fn pack_nibbles<D: Scalar>(vals: &[D]) -> Vec<u8> {
    assert!(
        vals.len().is_multiple_of(2),
        "a 4-bit stream needs an even (byte-aligned) element count, got {}",
        vals.len(),
    );
    let mut out = vec![0u8; vals.len() / 2];
    for (i, &v) in vals.iter().enumerate() {
        D::store(&mut out, i, v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A byte-multiple width serializes to its raw little-endian in-memory bytes, unchanged (the memcpy
    /// default).
    #[test]
    fn to_buf_byte_aligned_is_passthrough() {
        assert_eq!(i8::to_buf(&[1, -1, 127]), vec![1, 0xff, 127]);
        assert_eq!(
            f32::to_buf(&[1.0f32, 2.0]),
            [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat()
        );
        // The physical size is the `D::BITS` count in bytes.
        assert_eq!(f32::to_buf(&[0.0; 5]).len(), f32::size_in_bytes_from_length(5));
    }

    /// A 4-bit scalar packs two codes per byte, low element in the low nibble; half the length (the
    /// sub-byte override).
    #[test]
    fn to_buf_4bit_is_dense() {
        // `f4e2m1` codes 0x1, 0x2 -> one byte 0x21; 0x3, 0x4 -> 0x43.
        let vals: Vec<f4e2m1> = [0x1, 0x2, 0x3, 0x4].iter().map(|&b| f4e2m1::from_bits(b)).collect();
        assert_eq!(f4e2m1::to_buf(&vals), vec![0x21, 0x43]);
        assert_eq!(f4e2m1::to_buf(&vals).len(), f4e2m1::size_in_bytes_from_length(4));

        // `i4` packs the same way (low nibble of each element).
        let i4s: Vec<i4> = [1, 2, -1].iter().map(|&n| i4::from_i32(n)).collect();
        assert_eq!(i4::to_buf(&i4s[..2]), vec![0x21]);
    }
    /// The scalar `from_f32` / `from_i32` constructors are `const fn`: a `const` block evaluates them
    /// at compile time, and the compile-time result equals the runtime reference conversion byte for
    /// byte. This is the compile-time-fill foundation a typed `memset(value)` builds on (the fill
    /// value const-folds through these constructors).
    #[test]
    fn scalar_constructors_are_const() {
        const BF16_ONE: bf16 = bf16::from_f32(1.0);
        assert_eq!(BF16_ONE, bf16(half::bf16::from_f32(1.0)));
        assert_eq!(BF16_ONE.to_f32(), 1.0);

        const F8_ONE: f8e4m3 = f8e4m3::from_f32(1.0);
        assert_eq!(F8_ONE, f8e4m3(crate::float::f8_e4_from_f32(1.0)));

        const I4_NEG_ONE: i4 = i4::from_i32(-1);
        assert_eq!(I4_NEG_ONE.to_i32(), -1);
        const I4_SEVEN: i4 = i4::from_i32(7);
        assert_eq!(I4_SEVEN.to_i32(), 7);
    }

    /// `bf16::from_f32` delegates to `half::bf16::from_f32_const`, which `half`'s own runtime
    /// `from_f32` already forwards to, so the two are byte-identical across representative values.
    #[test]
    fn bf16_from_f32_matches_half_runtime() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 12.34, -56.78, 65504.0, f32::MIN, f32::MAX] {
            assert_eq!(bf16::from_f32(v), bf16(half::bf16::from_f32(v)), "mismatch at {v}");
        }
    }
}
