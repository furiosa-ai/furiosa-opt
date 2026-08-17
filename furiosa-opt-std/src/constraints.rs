//! Hardware constraints shared across engine tensor constructors.
//!
//! Each engine's `Tensor::new` validates its inputs against these. The values
//! and checks live here, in one place, so the rules are stated once and every
//! engine refers to the same definitions instead of repeating literals. Hardware parameters shared
//! with the published verifiers are re-exported from [`furiosa_opt_lower`] so a value is defined once
//! across both crates.
//!
//! Each check is a plain `fn` whose body is a single `const { ... }` block, so the compile-time
//! enforcement lives at the leaf and call sites stay plain calls (`assert_cluster_size::<Cluster>()`).
//! Because each check is its own `const` item, several violated checks all report in one compile
//! (a single shared `const {}` block would stop at the first panic).
//!
//! Interface (the checks call sites invoke) is at the top; the per-axis leaf helpers the
//! dimension-preserving checks build on, and the byte-size utility, are at the bottom.

use furiosa_mapping::M;

use crate::scalar::Scalar;

/// Bits in a byte / flit byte size / VRF capacity, single-sourced from the published verifier crate.
pub(crate) use furiosa_opt_lower::{BITS_PER_BYTE, FLIT_BYTES, VRF_BYTES};

/// Supported `Cluster` dimension sizes.
pub(crate) const CLUSTER_SIZES: [usize; 2] = [1, 2];

/// Byte granularity a packet must align to in the [`Fetch`](crate::engine) and
/// [`Switch`](crate::engine) engines, which pass packets through unchanged.
pub(crate) const PACKET_ALIGN_BYTES: usize = 8;

/// Supported `Slice` dimension sizes, in bytes.
pub(crate) const SLICE_SIZES: [usize; 3] = [64, 128, 256];

/// Asserts the `Cluster` dimension is one of the [`CLUSTER_SIZES`].
pub(crate) fn assert_cluster_size<Cluster: M>() {
    const {
        let mut i = 0;
        let mut ok = false;
        while i < CLUSTER_SIZES.len() {
            ok |= CLUSTER_SIZES[i] == Cluster::SIZE;
            i += 1;
        }
        assert!(ok, "Cluster size must be 1 or 2");
    };
}

/// Asserts the `Slice` dimension is one of the [`SLICE_SIZES`].
pub(crate) fn assert_slice_size<Slice: M>() {
    const {
        let mut i = 0;
        let mut ok = false;
        while i < SLICE_SIZES.len() {
            ok |= SLICE_SIZES[i] == Slice::SIZE;
            i += 1;
        }
        assert!(ok, "Slice size must be one of 64 | 128 | 256");
    };
}

/// Asserts a packet is a whole number of [`PACKET_ALIGN_BYTES`]-byte access words (Switch input).
pub(crate) fn assert_packet_aligned_by_access_width<D: Scalar, Packet: M>() {
    const {
        assert!(
            size_in_bytes(D::BITS, Packet::SIZE).is_multiple_of(PACKET_ALIGN_BYTES),
            "Packet size must be 8-byte aligned"
        );
    };
}

/// Asserts a packet is a non-zero multiple of [`PACKET_ALIGN_BYTES`] bytes, at most one flit
/// (Commit trim output: 8 | 16 | 24 | 32 bytes).
pub(crate) fn assert_packet_aligned_by_access_width_max_flit<D: Scalar, Packet: M>() {
    const {
        let bytes = size_in_bytes(D::BITS, Packet::SIZE);
        assert!(
            bytes != 0 && bytes.is_multiple_of(PACKET_ALIGN_BYTES) && bytes <= FLIT_BYTES,
            "Packet size must be 8, 16, 24, or 32 bytes"
        );
    };
}

/// Asserts a packet is exactly one flit ([`FLIT_BYTES`] bytes) (Collect, the compute engines, and
/// the Contraction lane output).
pub(crate) fn assert_packet_one_flit<D: Scalar, Packet: M>() {
    const {
        assert!(
            size_in_bytes(D::BITS, Packet::SIZE) == FLIT_BYTES,
            "Packet size must be exactly one flit (32 bytes)"
        );
    };
}

/// Asserts a packet is one or two flits (Contraction `outer` output: 32 or 64 bytes).
pub(crate) fn assert_packet_one_or_two_flit<D: Scalar, Packet: M>() {
    const {
        let bytes = size_in_bytes(D::BITS, Packet::SIZE);
        assert!(
            bytes == FLIT_BYTES || bytes == 2 * FLIT_BYTES,
            "Packet size must be one or two flits (32 or 64 bytes)"
        );
    };
}

/// Asserts one slice's vector register file operand fits [`VRF_BYTES`] (`to_vrf`).
///
/// A compile-time check, unlike the TRF's: the VRF is one undivided file per slice, so its capacity
/// is fixed here, while `to_trf` checks the whole tensor register file.
pub(crate) fn assert_vrf_capacity<D: Scalar, Element: M>() {
    const {
        assert!(
            size_in_bytes(D::BITS, Element::SIZE) <= VRF_BYTES,
            "VRF data must fit the vector register file (8192 bytes per slice)"
        );
    };
}

/// Asserts a DM → DM transfer preserves its whole `Chip` / `Cluster` / `Slice` partition.
///
/// Shared by every `DmTensor`(`View`/`ViewMut`) → `DmTensor`(`View`/`ViewMut`) primitive. Each leaf
/// check is its own `const` item, so a mismatch on any axis reports its own error (all in one
/// compile); an axis carried through unchanged passes its check trivially.
pub(crate) fn assert_dm_to_dm_dimension_preserved<Chip: M, Chip2: M, Cluster: M, Cluster2: M, Slice: M, Slice2: M>() {
    assert_chip_preserved::<Chip, Chip2>();
    assert_cluster_preserved::<Cluster, Cluster2>();
    assert_slice_preserved::<Slice, Slice2>();
}

/// Asserts a `reshape` preserves every one of its `Chip` / `Cluster` / `Slice` / `Element` partitions
/// (it relabels the mapping, it moves no data). Shares the per-axis leaf checks with
/// [`assert_dm_to_dm_dimension_preserved`].
pub(crate) fn assert_reshape_dimension_preserved<
    Chip: M,
    Chip2: M,
    Cluster: M,
    Cluster2: M,
    Slice: M,
    Slice2: M,
    Element: M,
    Element2: M,
>() {
    assert_chip_preserved::<Chip, Chip2>();
    assert_cluster_preserved::<Cluster, Cluster2>();
    assert_slice_preserved::<Slice, Slice2>();
    assert_element_preserved::<Element, Element2>();
}

/// Asserts a HBM `reshape` preserves its whole `Chip` / `Element` partition (it relabels the
/// mapping, it moves no data). HBM has no `Cluster`/`Slice` axis, so those two are the whole
/// partition here.
pub(crate) fn assert_hbm_reshape_dimension_preserved<Chip: M, Chip2: M, Element: M, Element2: M>() {
    assert_chip_preserved::<Chip, Chip2>();
    assert_element_preserved::<Element, Element2>();
}

/// Asserts the `Chip` size is preserved across a transfer / reshape.
fn assert_chip_preserved<Chip: M, Chip2: M>() {
    const { assert!(Chip::SIZE == Chip2::SIZE, "Chip size must be preserved") };
}

/// Asserts the `Cluster` size is preserved across a transfer / reshape.
fn assert_cluster_preserved<Cluster: M, Cluster2: M>() {
    const { assert!(Cluster::SIZE == Cluster2::SIZE, "Cluster size must be preserved") };
}

/// Asserts the `Slice` size is preserved across a transfer / reshape.
fn assert_slice_preserved<Slice: M, Slice2: M>() {
    const { assert!(Slice::SIZE == Slice2::SIZE, "Slice size must be preserved") };
}

/// Asserts the `Element` size is preserved across a transfer / reshape.
fn assert_element_preserved<Element: M, Element2: M>() {
    const { assert!(Element::SIZE == Element2::SIZE, "Element size must be preserved") };
}

/// Byte size of `length` elements of a `bits`-wide scalar.
///
/// `const fn` twin of [`Scalar::size_in_bytes_from_length`](crate::scalar::Scalar::size_in_bytes_from_length),
/// callable from the `const { ... }` constraint checks. Panics if the total bit count is not byte-aligned.
pub(crate) const fn size_in_bytes(bits: usize, length: usize) -> usize {
    assert!(
        (length * bits).is_multiple_of(BITS_PER_BYTE),
        "total bits must be byte-aligned"
    );
    (length * bits) / BITS_PER_BYTE
}
