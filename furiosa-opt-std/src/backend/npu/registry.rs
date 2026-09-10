//! The binary's device function registry.
//!
//! How it gets there: the driver rides the build as cargo's `RUSTC_WORKSPACE_WRAPPER`. While
//! the binary compiles, the driver lowers every `#[device]` instantiation the binary launches
//! and appends one object file to the binary's own link line (`link.rs` in npu-opt, the wire
//! format's other end). That object's only content is a `furiosa_kernels` section holding the
//! serialized entries. The linker keeps the section in the final binary and defines
//! `__start_`/`__stop_` symbols at its bounds.
//!
//! What is in it: bincode of `Vec<(fn path, Key, image)>`, ordered by (fn path, key):
//! - fn path: `crate::module::fn`, exactly what the macro derives at the launch site
//!   (`module_path!()` plus the fn name)
//! - [`Key`]: the instantiation's numeric generic arguments
//! - image: the function's image ([`Bytes`]), parsed by `Function::load`
//!
//! How it is read: the first launch takes the section as one static slice (the linker-defined
//! bound symbols: no file, no env, no I/O), deserializes it once, and each `#[device]` fn's
//! [`entries`] filters by its own fn path. The launch then selects by key.

use std::sync::OnceLock;

/// A launched instantiation's identity: the numeric generic arguments in declaration order, an
/// `AxisName` type param's `SIZE` or a `usize` const param's value. A concrete fn's key is
/// empty. Any other parameter kind fails to compile under `backend = "npu"` (the macro's key
/// expression requires exactly these two).
pub type Key = Vec<u64>;

/// A registry entry's image, parsed by `Function::load`.
pub type Bytes = &'static [u8];

/// This fn's registry entries: one `(key, image)` per launched instantiation.
pub(crate) fn entries(path: &str) -> impl Iterator<Item = (&'static Key, Bytes)> {
    static REGISTRY: OnceLock<Vec<(&str, Key, &[u8])>> = OnceLock::new();
    REGISTRY
        .get_or_init(|| {
            std::hint::black_box(&raw const ANCHOR);
            let start = &raw const START;
            let stop = &raw const STOP;
            // SAFETY: the linker-defined bounds delimit the loaded `furiosa_kernels` section.
            match unsafe { std::slice::from_raw_parts(start, stop.addr() - start.addr()) } {
                [] => Vec::new(),
                bytes => bincode::deserialize(bytes).expect("malformed device function registry"),
            }
        })
        .iter()
        .filter(move |(entry, ..)| *entry == path)
        .map(|(_, key, bytes)| (key, *bytes))
}

// NOTE: the section name, its `__start_`/`__stop_` bound symbols, and the bincode row format
// are one wire contract with the driver's link pass (`link.rs` in npu-opt). The npu-opt
// registry test reads a real binary end-to-end and fails on a drift on either side. The name
// must be a valid C identifier (`__start_` synthesis requires it) and must keep the vendor
// prefix: section names share one global namespace with every linked object, C libraries
// included, and the linker concatenates same-named sections from all of them.
unsafe extern "C" {
    #[link_name = "__start_furiosa_kernels"]
    static START: u8;
    #[link_name = "__stop_furiosa_kernels"]
    static STOP: u8;
}

/// Keeps the section and its bound symbols present in a link the driver never saw (a plain
/// `cargo build`, a doctest). Such a binary panics at launch, not at link.
#[used]
#[unsafe(link_section = "furiosa_kernels")]
static ANCHOR: [u8; 0] = [];
