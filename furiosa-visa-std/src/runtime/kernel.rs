use furiosa_mapping::M;

use super::ffi;
use crate::memory_tensor::HbmTensor;
use crate::prelude::{Address, HostTensor};
use crate::scalar::{Opt, Scalar};

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

    fn cpu(size: usize) -> Self {
        let ptr = unsafe { ffi::lib().furiosa_buffer_cpu(size) };
        assert!(!ptr.is_null(), "failed to allocate CPU buffer");
        Buffer(ptr)
    }

    fn npu(addr: u64, len: usize) -> Self {
        Buffer(unsafe { ffi::lib().furiosa_buffer_from_npu(ffi::rt(), addr, len) })
    }

    fn data_ptr(&self) -> *mut u8 {
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

    /// Copies host tensor to device via DMA.
    pub async fn write<D: Scalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element>,
        addr: Address,
    ) -> HbmTensor<D, Chip, Element2> {
        let stride = std::mem::size_of::<D>();
        let len = host.data().len() * stride;
        log::debug!("write: addr=0x{addr:x}, len={len}");

        let src = Buffer::cpu(len);
        let ptr = src.data_ptr();
        for (i, opt) in host.data().iter().enumerate() {
            let offset = i * stride;
            unsafe {
                match opt {
                    Opt::Init(val) => {
                        std::ptr::copy_nonoverlapping(val as *const D as *const u8, ptr.add(offset), stride);
                    }
                    Opt::Uninit => {
                        std::ptr::write_bytes(ptr.add(offset), 0, stride);
                    }
                }
            }
        }

        let dst = Buffer::npu(addr, len);
        assert!(
            unsafe { ffi::lib().furiosa_write(ffi::rt(), src.as_ptr(), dst.as_ptr()) } == 0,
            "DMA write failed"
        );
        unsafe { HbmTensor::from_addr(addr) }
    }

    /// Copies device tensor to host via DMA.
    pub async fn read<D: Scalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element>,
    ) -> HostTensor<D, Element2> {
        let stride = std::mem::size_of::<D>();
        let len = hbm.data().len() * stride;
        log::debug!("read: addr=0x{:x}, len={len}", hbm.address());

        let src = Buffer::npu(hbm.address(), len);
        let dst = Buffer::cpu(len);
        let count = hbm.data().len();
        assert!(
            unsafe { ffi::lib().furiosa_read(ffi::rt(), src.as_ptr(), dst.as_ptr()) } == 0,
            "DMA read failed"
        );
        let ptr = dst.data_ptr() as *const D;
        let elems: Vec<Opt<D>> = (0..count)
            .map(|i| Opt::Init(unsafe { std::ptr::read(ptr.add(i)) }))
            .collect();
        HostTensor::from_buf(elems)
    }

    /// Execute kernel.
    pub async fn run(&self, inputs: &[Buffer], outputs: &[Buffer]) {
        log::debug!("run: inputs={}, outputs={}", inputs.len(), outputs.len());
        let in_ptrs: Vec<*const ffi::Buffer> = inputs.iter().map(|b| b.as_ptr()).collect();
        let out_ptrs: Vec<*const ffi::Buffer> = outputs.iter().map(|b| b.as_ptr()).collect();
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
        Buffer(ptr)
    }
}
