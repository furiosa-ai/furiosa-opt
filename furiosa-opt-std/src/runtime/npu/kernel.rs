use furiosa_mapping::M;

use crate::prelude::HostTensor;
use crate::scalar::Scalar;
use crate::tensor::memory::{Address, HbmTensor};

use super::Npu;
use super::ffi;

/// Opaque handle to a device-runtime buffer.
///
/// Wraps a raw pointer returned by the FFI layer. Automatically freed on drop.
#[derive(Debug)]
pub struct Buffer(*mut ffi::Buffer);

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::lib().furiosa_buffer_free(self.0) }
        }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Buffer(unsafe { ffi::lib().furiosa_buffer_clone(self.0) })
    }
}

impl Buffer {
    pub(super) fn from_raw(ptr: *mut ffi::Buffer) -> Self {
        Buffer(ptr)
    }

    pub(super) fn as_ptr(&self) -> *const ffi::Buffer {
        self.0
    }

    pub(crate) fn cpu(size: usize) -> Self {
        let ptr = unsafe { ffi::lib().furiosa_buffer_cpu(size) };
        assert!(!ptr.is_null(), "failed to allocate CPU buffer");
        Buffer::from_raw(ptr)
    }

    pub(crate) fn npu(addr: u64, len: usize) -> Self {
        Buffer::from_raw(unsafe { ffi::lib().furiosa_buffer_from_npu(ffi::rt(), addr, len) })
    }

    pub(crate) fn data_ptr(&self) -> *mut u8 {
        unsafe { ffi::lib().furiosa_buffer_addr(self.as_ptr()) as *mut u8 }
    }
}

/// Device kernel loaded from a serialized binary.
pub struct Kernel {
    ptr: *mut ffi::Kernel,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel").finish_non_exhaustive()
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { ffi::lib().furiosa_kernel_free(self.ptr) }
    }
}

impl Kernel {
    /// Load kernel from serialized binary. On miss, defers to [`crate::diag`] for structured panic rendering;
    /// the runtime itself does no formatting or log parsing.
    pub async fn load(path: &str) -> Self {
        crate::diag::install_hook();
        let Ok(data) = std::fs::read(path) else {
            match crate::diag::failure_payload(path) {
                Some(payload) => panic!("{payload}"),
                None => panic!("failed to load kernel `{path}`"),
            }
        };
        log::debug!("load: {} bytes", data.len());

        let ptr = unsafe { ffi::lib().furiosa_kernel_load(ffi::rt(), data.as_ptr(), data.len()) };
        assert!(!ptr.is_null(), "failed to load kernel");

        log::debug!("load: kernel loaded");
        Kernel { ptr }
    }

    /// Execute kernel.
    pub async fn run(&self, inputs: &[Buffer], outputs: &[Buffer]) {
        log::debug!("run: inputs={}, outputs={}", inputs.len(), outputs.len());
        let in_ptrs = inputs.iter().map(|b| b.as_ptr()).collect::<Vec<_>>();
        let out_ptrs = outputs.iter().map(|b| b.as_ptr()).collect::<Vec<_>>();
        assert!(
            unsafe {
                ffi::lib().furiosa_kernel_run(
                    self.ptr,
                    ffi::rt(),
                    in_ptrs.as_ptr(),
                    in_ptrs.len(),
                    out_ptrs.as_ptr(),
                    out_ptrs.len(),
                )
            } == 0,
            "kernel execution failed"
        );
    }

    /// Allocate a buffer on the device.
    pub fn alloc(&self, size: usize) -> Buffer {
        let ptr = unsafe { ffi::lib().furiosa_buffer_npu(ffi::rt(), size) };
        assert!(!ptr.is_null(), "failed to allocate buffer");
        Buffer::from_raw(ptr)
    }

    /// Copies a host staging tensor to NPU HBM via DMA.
    ///
    /// This is host-side I/O, not tensor computation. The copied bytes become the device
    /// function's HBM input; value-producing tensor operations are performed on the NPU.
    pub async fn write<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Npu>,
        addr: Address,
    ) -> HbmTensor<D, Chip, Element2, Npu> {
        let stride = std::mem::size_of::<D>();
        let buf = host.to_buf();
        let len = buf.len() * stride;
        log::debug!("write: addr=0x{addr:x}, len={len}");

        let src = Buffer::cpu(len);
        let ptr = src.data_ptr();
        for (i, value) in buf.iter().enumerate() {
            // SAFETY: `ptr` points to `len` writable bytes; each copy writes one `D`.
            unsafe {
                std::ptr::copy_nonoverlapping(value as *const D as *const u8, ptr.add(i * stride), stride);
            }
        }

        let dst = Buffer::npu(addr, len);
        assert!(
            unsafe { ffi::lib().furiosa_write(ffi::rt(), src.as_ptr(), dst.as_ptr()) } == 0,
            "DMA write failed"
        );
        unsafe { HbmTensor::from_addr(addr) }
    }

    /// Copies an NPU HBM tensor back into a host staging tensor via DMA.
    ///
    /// The returned `HostTensor` owns the bytes read from device memory as native `Vec<D>` data so
    /// `to_buf` can expose them without an `Opt<D>` conversion layer.
    pub async fn read<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Npu>,
    ) -> HostTensor<D, Element2, Npu> {
        let stride = std::mem::size_of::<D>();
        let count = furiosa_mapping::Pair::<Chip, Element>::SIZE;
        let len = count * stride;
        log::debug!("read: addr=0x{:x}, len={len}", hbm.address());

        let src = Buffer::npu(hbm.address(), len);
        let dst = Buffer::cpu(len);
        assert!(
            unsafe { ffi::lib().furiosa_read(ffi::rt(), src.as_ptr(), dst.as_ptr()) } == 0,
            "DMA read failed"
        );
        let ptr = dst.data_ptr() as *const u8;
        HostTensor::from_buf((0..count).map(|i| {
            // SAFETY: `ptr` points to `count * stride` readable bytes from the DMA copy.
            unsafe { std::ptr::read(ptr.add(i * stride) as *const D) }
        }))
    }
}

/// Builds the on-disk `.bin` path for a kernel, from the perspective of the `#[device]` macro expansion.
/// `module_path!()` includes the crate name (`my_crate::sub::mod`), but the plugin's artifact names use
/// `tcx.def_path_str` which strips it (`sub::mod`); this helper drops the leading crate segment and mangles
/// `::` → `__` so the stem agrees with `rustc_plugin::build_one`.
pub fn kernel_path(out_dir: &str, pkg: &str, module_path: &str, fn_name: &str) -> String {
    let stem = module_path
        .split("::")
        .chain(std::iter::once(fn_name))
        .skip(1)
        .collect::<Vec<_>>()
        .join("__");
    format!("{out_dir}/{pkg}/{stem}.bin")
}
