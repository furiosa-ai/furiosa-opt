use std::fmt;
use std::ptr;
use std::slice;

use furiosa_mapping::M;

use crate::prelude::HostTensor;
use crate::scalar::MaterializableScalar;
use crate::storage::BufStorage;
use crate::tensor::Tensor;
use crate::tensor::memory::HbmTensor;

use super::Npu;
use super::ffi;

/// Opaque handle to a device-runtime buffer.
///
/// Wraps a raw pointer returned by the FFI layer. Automatically freed on drop.
#[derive(Debug)]
pub struct Buffer(*mut ffi::NpuBuffer);

// SAFETY: The runtime owns synchronization for this opaque device handle, which may move across threads.
unsafe impl Send for Buffer {}
// SAFETY: Shared access exposes only runtime operations that accept an immutable buffer handle.
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: This handle came from the runtime and this Drop is its unique owner.
            unsafe { ffi::furiosa_npu_buffer_free(self.0) }
        }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        // SAFETY: The runtime clone accepts a live handle and returns an independently owned handle.
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
        // SAFETY: The caller supplies a device address and length for the synchronous runtime operation.
        Buffer::from_raw(unsafe { ffi::furiosa_npu_buffer_from(ffi::rt(), addr, len) })
    }

    pub(crate) fn alloc(size: usize) -> Self {
        // SAFETY: The runtime allocator accepts any byte length and returns an owned opaque handle.
        let ptr = unsafe { ffi::furiosa_npu_buffer(ffi::rt(), size) };
        assert!(!ptr.is_null(), "failed to allocate buffer");
        Buffer::from_raw(ptr)
    }

    pub(crate) fn offset(&self) -> u64 {
        // SAFETY: `self.0` is a live runtime buffer handle for the lifetime of `self`.
        unsafe { ffi::furiosa_npu_buffer_offset(self.0) }
    }
}

/// An owned DMA-heap allocation managed by the device runtime.
pub struct CpuBuffer {
    ptr: *mut ffi::CpuBuffer,
    len: usize,
}

// SAFETY: The handle uniquely owns a non-thread-affine DMA allocation and may move across threads.
unsafe impl Send for CpuBuffer {}
// SAFETY: Shared access exposes only immutable bytes; mutation requires unique `&mut` access.
unsafe impl Sync for CpuBuffer {}

impl fmt::Debug for CpuBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBuffer").field("len", &self.len()).finish()
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: This handle came from `furiosa_cpu_buffer` and this Drop is its unique owner.
            unsafe { ffi::furiosa_cpu_buffer_free(self.ptr) }
        }
    }
}

impl From<Vec<u8>> for CpuBuffer {
    fn from(v: Vec<u8>) -> Self {
        Self::from_slice(&v)
    }
}

impl From<CpuBuffer> for Vec<u8> {
    fn from(c: CpuBuffer) -> Self {
        c.as_slice().to_vec()
    }
}

impl AsRef<[u8]> for CpuBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for CpuBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Clone for CpuBuffer {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl CpuBuffer {
    fn alloc(len: usize) -> Self {
        // SAFETY: The runtime allocator accepts a byte length and returns an owned DMA-heap handle.
        let ptr = unsafe { ffi::furiosa_cpu_buffer(len) };
        assert!(!ptr.is_null(), "failed to allocate DMA buffer");
        CpuBuffer { ptr, len }
    }

    fn from_slice(bytes: &[u8]) -> Self {
        let cpu = Self::alloc(bytes.len());
        if !bytes.is_empty() {
            // SAFETY: Both regions are valid for `bytes.len()` bytes and belong to distinct allocations.
            unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), cpu.data_ptr(), bytes.len()) };
        }
        cpu
    }

    fn as_ptr(&self) -> *const ffi::CpuBuffer {
        self.ptr
    }

    fn data_ptr(&self) -> *mut u8 {
        // SAFETY: `self.ptr` is a live CPU buffer handle whose address remains valid until Drop.
        unsafe { ffi::furiosa_cpu_buffer_addr(self.ptr) as *mut u8 }
    }

    /// The allocated byte length, tracked here (like `Vec`) rather than queried via
    /// `furiosa_cpu_buffer_len`, whose result does not match the requested size and breaks slice bounds.
    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: The runtime allocation covers `len` initialized bytes and lives as long as `self`.
        unsafe { slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Unique access covers the runtime allocation's full initialized byte range.
        unsafe { slice::from_raw_parts_mut(self.data_ptr(), self.len()) }
    }
}

/// Device kernel loaded from a serialized binary.
pub struct Kernel {
    ptr: *mut ffi::Kernel,
}

// SAFETY: The runtime kernel handle is not thread-affine and all operations synchronize in the runtime.
unsafe impl Send for Kernel {}
// SAFETY: Shared access invokes only runtime functions that accept an immutable kernel handle.
unsafe impl Sync for Kernel {}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel").finish_non_exhaustive()
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is the live handle uniquely owned by this Kernel.
        unsafe { ffi::furiosa_kernel_free(self.ptr) }
    }
}

impl Kernel {
    /// Load a kernel from bytes embedded in the binary (see the `#[device]` macro).
    pub async fn load(data: &[u8]) -> Self {
        assert!(!data.is_empty(), "attempted to load an uncompiled NPU kernel");
        log::debug!("load: {} bytes", data.len());
        // SAFETY: `data` remains valid for the synchronous load and the runtime copies the kernel image.
        let ptr = unsafe { ffi::furiosa_kernel_load(ffi::rt(), data.as_ptr(), data.len()) };
        assert!(!ptr.is_null(), "failed to load kernel");
        Kernel { ptr }
    }

    /// Execute the kernel. When a `tracing` subscriber listens on the `span::npu`
    /// target, profiling is armed and each decoded TUC span is emitted off the
    /// launch hot path as an `info_span!` (see [`ffi::run`]).
    pub async fn run(&self, inputs: &[Buffer], outputs: &[Buffer]) {
        log::debug!("run: inputs={}, outputs={}", inputs.len(), outputs.len());
        let in_ptrs = inputs.iter().map(|b| b.as_ptr()).collect::<Vec<_>>();
        let out_ptrs = outputs.iter().map(|b| b.as_ptr()).collect::<Vec<_>>();
        assert!(ffi::run(self.ptr, &in_ptrs, &out_ptrs) == 0, "kernel execution failed");
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
        let src = host.storage().inner();
        let len = src.len();
        let dst = Buffer::alloc(len);
        let addr = dst.offset();
        log::debug!("write: addr=0x{addr:x}, len={len}");
        assert!(
            // SAFETY: Both owned runtime buffers remain live for the synchronous DMA operation.
            unsafe { ffi::furiosa_write(ffi::rt(), src.as_ptr(), dst.as_ptr()) } == 0,
            "DMA write failed"
        );
        // SAFETY: `addr` names `dst`, whose ownership is attached to the returned tensor.
        unsafe { HbmTensor::from_addr(addr) }.owns(dst)
    }

    /// Copies an NPU HBM tensor back into a host staging tensor via DMA.
    ///
    /// The returned `HostTensor` owns the DMA allocation populated by the device read.
    pub async fn read<D: MaterializableScalar, Chip: M, Element: M, Element2: M>(
        hbm: &HbmTensor<D, Chip, Element, Npu>,
    ) -> HostTensor<D, Element2, Npu> {
        let count = furiosa_mapping::Pair::<Chip, Element>::SIZE;
        let len = D::size_in_bytes_from_length(count);
        let hbm_addr = hbm.address();
        log::debug!("read: addr=0x{:x}, len={len}", hbm_addr);

        let src = Buffer::npu(hbm_addr, len);
        let cpu = CpuBuffer::alloc(len);
        assert!(
            // SAFETY: Both owned runtime buffers remain live for the synchronous DMA operation.
            unsafe { ffi::furiosa_read(ffi::rt(), src.as_ptr(), cpu.as_ptr()) } == 0,
            "DMA read failed"
        );
        let storage = BufStorage::<D, CpuBuffer>::from(cpu);
        Tensor::from_inner(storage).into()
    }
}
