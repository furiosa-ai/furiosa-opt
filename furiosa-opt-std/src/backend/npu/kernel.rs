use furiosa_mapping::M;

use crate::prelude::HostTensor;
use crate::scalar::MaterializableScalar;
use crate::tensor::memory::HbmTensor;

use super::Npu;
use super::ffi;

/// Opaque handle to a device-runtime buffer.
///
/// Wraps a raw pointer returned by the FFI layer. Automatically freed on drop.
#[derive(Debug)]
pub struct Buffer(*mut ffi::NpuBuffer);

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::furiosa_npu_buffer_free(self.0) }
        }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Buffer(unsafe { ffi::furiosa_npu_buffer_clone(self.0) })
    }
}

impl Buffer {
    pub(super) fn from_raw(ptr: *mut ffi::NpuBuffer) -> Self {
        Buffer(ptr)
    }

    pub(super) fn as_ptr(&self) -> *const ffi::NpuBuffer {
        self.0
    }

    pub(crate) fn npu(addr: u64, len: usize) -> Self {
        Buffer::from_raw(unsafe { ffi::furiosa_npu_buffer_from(ffi::rt(), addr, len) })
    }

    pub(crate) fn alloc(size: usize) -> Self {
        let ptr = unsafe { ffi::furiosa_npu_buffer(ffi::rt(), size) };
        assert!(!ptr.is_null(), "failed to allocate buffer");
        Buffer::from_raw(ptr)
    }

    pub(crate) fn offset(&self) -> u64 {
        unsafe { ffi::furiosa_npu_buffer_offset(self.0) }
    }
}

struct CpuBuffer(*mut ffi::CpuBuffer);

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::furiosa_cpu_buffer_free(self.0) }
        }
    }
}

impl CpuBuffer {
    fn cpu(size: usize) -> Self {
        let ptr = unsafe { ffi::furiosa_cpu_buffer(size) };
        assert!(!ptr.is_null(), "failed to allocate CPU buffer");
        CpuBuffer(ptr)
    }

    fn as_ptr(&self) -> *const ffi::CpuBuffer {
        self.0
    }

    fn data_ptr(&self) -> *mut u8 {
        unsafe { ffi::furiosa_cpu_buffer_addr(self.as_ptr()) as *mut u8 }
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
        unsafe { ffi::furiosa_kernel_free(self.ptr) }
    }
}

impl Kernel {
    /// Load a kernel from bytes embedded in the binary (see the `#[device]` macro).
    pub async fn load(data: &[u8]) -> Self {
        assert!(!data.is_empty(), "attempted to load an uncompiled NPU kernel");
        log::debug!("load: {} bytes", data.len());
        let ptr = unsafe { ffi::furiosa_kernel_load(ffi::rt(), data.as_ptr(), data.len()) };
        assert!(!ptr.is_null(), "failed to load kernel");
        Kernel { ptr }
    }

    /// Execute kernel.
    pub async fn run(&self, inputs: &[Buffer], outputs: &[Buffer]) {
        log::debug!("run: inputs={}, outputs={}", inputs.len(), outputs.len());
        let in_ptrs = inputs.iter().map(|b| b.as_ptr()).collect::<Vec<_>>();
        let out_ptrs = outputs.iter().map(|b| b.as_ptr()).collect::<Vec<_>>();
        assert!(
            unsafe {
                ffi::furiosa_kernel_run(
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
        Buffer::alloc(size)
    }

    /// Copies a host staging tensor to NPU HBM via DMA.
    ///
    /// This is host-side I/O, not tensor computation. The copied bytes become the device
    /// function's HBM input; value-producing tensor operations are performed on the NPU.
    pub async fn write<D: MaterializableScalar, Element: M, Chip: M, Element2: M>(
        host: &HostTensor<D, Element, Npu>,
    ) -> HbmTensor<D, Chip, Element2, Npu> {
        let stride = std::mem::size_of::<D>();
        let buf = host.clone().into_vec();
        let len = buf.len() * stride;

        let src = CpuBuffer::cpu(len);
        let ptr = src.data_ptr();
        for (i, value) in buf.iter().enumerate() {
            // SAFETY: `ptr` points to `len` writable bytes; each copy writes one `D`.
            unsafe {
                std::ptr::copy_nonoverlapping(value as *const D as *const u8, ptr.add(i * stride), stride);
            }
        }

        let dst = Buffer::alloc(len);
        let addr = dst.offset();
        log::debug!("write: addr=0x{addr:x}, len={len}");
        assert!(
            unsafe { ffi::furiosa_write(ffi::rt(), src.as_ptr(), dst.as_ptr()) } == 0,
            "DMA write failed"
        );
        unsafe { HbmTensor::from_addr(addr) }.owns(dst)
    }

    /// Copies an NPU HBM tensor back into a host staging tensor via DMA.
    ///
    /// The returned `HostTensor` owns the bytes read from device memory as native `Vec<D>` data so
    /// `to_vec` can expose them without an `Opt<D>` conversion layer.
    pub async fn read<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Npu>,
    ) -> HostTensor<D, Element2, Npu> {
        let stride = std::mem::size_of::<D>();
        let count = furiosa_mapping::Pair::<Chip, Element>::SIZE;
        let len = count * stride;
        let hbm_addr = hbm.address();
        log::debug!("read: addr=0x{:x}, len={len}", hbm_addr);

        let src = Buffer::npu(hbm_addr, len);
        let dst = CpuBuffer::cpu(len);
        assert!(
            unsafe { ffi::furiosa_read(ffi::rt(), src.as_ptr(), dst.as_ptr()) } == 0,
            "DMA read failed"
        );
        let ptr = dst.data_ptr() as *const u8;
        HostTensor::from_vec((0..count).map(|i| {
            // SAFETY: `ptr` points to `count * stride` readable bytes from the DMA copy.
            unsafe { std::ptr::read(ptr.add(i * stride) as *const D) }
        }))
    }
}
