//! Memory-mapped registers. Naming one is the only unsafe step; from then on it is read and
//! written like a cell. A hardware unit lists its registers as constants of this type, so the
//! `unsafe` that says "this is a register, mapped for as long as I run" appears once per register
//! and nowhere in the unit's logic. The host names registers the same way through a mapping.

use core::ptr;

/// One register of `T`, accessed volatile.
#[derive(Clone, Copy, Debug)]
pub struct Reg<T>(*mut T);

// SAFETY: a register has no memory of its own to race on; the hardware serializes accesses.
unsafe impl<T> Sync for Reg<T> {}
// SAFETY: as above; a register may be used from whichever thread holds its owner.
unsafe impl<T> Send for Reg<T> {}

impl<T: Copy> Reg<T> {
    /// # Safety
    ///
    /// `register` must point at a `T`-sized register mapped for as long as the value is used.
    pub const unsafe fn new(register: *mut T) -> Self {
        Self(register)
    }

    /// # Safety
    ///
    /// `address` must be a `T`-sized register the MMU maps for as long as the image runs.
    pub const unsafe fn at(address: usize) -> Self {
        Self(ptr::without_provenance_mut(address))
    }

    pub fn read(&self) -> T {
        // SAFETY: `at` promised a mapped register of this width.
        unsafe { ptr::read_volatile(self.0) }
    }

    pub fn write(&self, value: T) {
        // SAFETY: `at` promised a mapped register of this width.
        unsafe { ptr::write_volatile(self.0, value) }
    }

    /// The register `index` entries of `T` past this one: a row of a register array.
    ///
    /// # Safety
    ///
    /// The array must extend that far.
    pub const unsafe fn offset(&self, index: usize) -> Self {
        // SAFETY: the caller promised the array reaches `index`.
        Self(unsafe { self.0.add(index) })
    }
}

/// Spins until `poll` is ready. The hardware units answer `Poll`, and a caller with nothing else
/// to do waits like this; one with something else to do polls itself.
pub fn block_on<T>(mut poll: impl FnMut() -> core::task::Poll<T>) -> T {
    loop {
        if let core::task::Poll::Ready(value) = poll() {
            return value;
        }
        core::hint::spin_loop();
    }
}
