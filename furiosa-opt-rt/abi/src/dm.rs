//! Resident SRAM window bound by the host runtime.
//!
//! Compiler temporaries grow from zero; residents grow down from SRAM end.
//! Admission keeps temporaries below every live resident.

use core::ops::Range;

/// Bytes per SRAM page on one slice.
pub const PAGE_SIZE: u64 = 32 * 1024;

/// Physical pages a resident may occupy: every page of a slice. What keeps a launch's temporaries
/// out of them is the admission check, not a page held back for them.
pub const RESIDENT_PAGES: Range<u64> = 0..16;

/// Virtual pages that map one-to-one onto [`RESIDENT_PAGES`]: the top of the 128-entry table.
pub const VIRTUAL_PAGES: Range<u64> = 112..128;

/// First physical byte a resident may occupy.
pub const RESIDENT_BASE: u64 = RESIDENT_PAGES.start * PAGE_SIZE;

/// One past the last physical byte a resident may occupy.
pub const RESIDENT_END: u64 = RESIDENT_PAGES.end * PAGE_SIZE;

/// Virtual byte the kernel uses to address [`RESIDENT_BASE`].
pub const VIRTUAL_BASE: u64 = VIRTUAL_PAGES.start * PAGE_SIZE;

/// Alignment of a resident's base address.
pub const ALIGNMENT: u64 = 256;

/// Virtual address of a byte in the resident physical window.
pub const fn virtual_addr(physical: u64) -> u64 {
    assert!(physical < RESIDENT_END);
    physical - RESIDENT_BASE + VIRTUAL_BASE
}

/// Physical address of a byte in the resident virtual window.
pub const fn physical_addr(virtual_addr: u64) -> u64 {
    assert!(virtual_addr >= VIRTUAL_BASE && virtual_addr < VIRTUAL_PAGES.end * PAGE_SIZE);
    virtual_addr - VIRTUAL_BASE + RESIDENT_BASE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_covers_residents() {
        assert_eq!(
            VIRTUAL_PAGES.end - VIRTUAL_PAGES.start,
            RESIDENT_PAGES.end - RESIDENT_PAGES.start
        );
        assert_eq!(
            VIRTUAL_BASE - RESIDENT_BASE,
            (VIRTUAL_PAGES.start - RESIDENT_PAGES.start) * PAGE_SIZE
        );
    }
}
