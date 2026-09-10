//! Authenticates and lands the firmware image staged by the host.
//!
//! - Its signed digest binds one separately built firmware image.
//! - MMU and caches off: every address is physical.
//! - Speaks only `furiosa_opt_abi::bootloader` and the driver's rings.

#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use core::ptr;

use furiosa_opt_abi::bootloader::{Code, IMAGE_BASE, REPLY_WORDS, REQUEST_WORDS, header, in_range};
use furiosa_opt_abi::reg::{Reg, block_on};
use furiosa_opt_abi::ring::{Consumer, Producer, Ring};
use furiosa_opt_bootloader::Image;

const FIRMWARE_DIGEST: &[u8; 32] = include_bytes!(concat!(env!("OUT_DIR"), "/firmware.sha256"));

global_asm!(
    r#"
    .section .text.reset_vector, "ax"
    .global _reset_entry
    .type _reset_entry, %function
_reset_entry:
    msr daifset, #0xf
    mov x0, #0x480
    msr scr_el3, x0
    ldr x0, =exception_vectors
    msr vbar_el1, x0
    mov x0, #0x30
    lsl x0, x0, #16
    msr cpacr_el1, x0
    ldr x0, =25000000
    msr cntfrq_el0, x0
    ldr x0, =__boot_stack_end
    msr sp_el1, x0
    ldr x0, =__bss_start__
    ldr x1, =__bss_end__
3:
    cmp x0, x1
    b.hs 4f
    str xzr, [x0], #8
    b 3b
4:
    mov x0, #0x3c5
    msr spsr_el3, x0
    ldr x0, =boot_main
    msr elr_el3, x0
    eret

    .section .text.vector, "ax"
    .balign 2048
exception_vectors:
    .rept 16
    b exception_halt
    .space 124
    .endr
exception_halt:
    msr daifset, #0xf
2:
    wfe
    b 2b
"#
);

const DOORBELL_ENTRIES: usize = 0x40_0000;
const DOORBELL_STRIDE: usize = 0x400;
const DOORBELL_PRODUCER: usize = 0x80_0800;
const DOORBELL_CONSUMER: usize = 0x80_0900;
// SAFETY: the standalone memory map fixes these registers for the bootloader's lifetime.
const DOORBELL_STATUS: Reg<u64> = unsafe { Reg::at(0x80_0cd8) };
const DOORBELL_COUNT: usize = 64;
const DOORBELL_CAPACITY: usize = 127;
// SAFETY: the completion ring's registers and its 255 entries, fixed by the same map.
const COMPLETION: Ring = unsafe {
    Ring::new(
        Reg::at(0x50_0020_0008),
        Reg::at(0x50_0020_0000),
        Reg::at(0x50_0020_0004),
        255,
    )
};

#[unsafe(no_mangle)]
extern "C" fn boot_main() -> ! {
    loop {
        let status = DOORBELL_STATUS.read();
        for slot in (0..DOORBELL_COUNT).filter(|slot| status & (1 << slot) != 0) {
            serve(slot);
        }
        core::hint::spin_loop();
    }
}

/// Answers whatever complete request waits in doorbell `slot`, and jumps when it was a boot.
fn serve(slot: usize) {
    // SAFETY: doorbell `slot`'s entries and index registers, fixed by the standalone memory map.
    let ring = unsafe {
        Ring::new(
            Reg::at(DOORBELL_ENTRIES + slot * DOORBELL_STRIDE),
            Reg::at(DOORBELL_PRODUCER + slot * 4),
            Reg::at(DOORBELL_CONSUMER + slot * 4),
            DOORBELL_CAPACITY,
        )
    };
    let Ok(doorbell) = Consumer::new(ring) else {
        return;
    };
    let read = |offset| {
        let mut word = [0];
        doorbell.copy(offset, &mut word).map(|()| word[0])
    };
    let consume = |words| {
        let _ = doorbell.consume(words);
    };
    let Ok(available) = doorbell.available() else {
        return;
    };
    if available == 0 {
        return;
    }
    let Some((id, words)) = read(0).ok().and_then(header::unpack) else {
        consume(1);
        return;
    };
    let words = usize::from(words);
    if available <= words {
        return;
    }
    if words != REQUEST_WORDS {
        consume(1 + words);
        complete(id, Code::NotBoot);
        return;
    }
    let (Ok(addr), Ok(len), Ok(token)) = (read(1), read(2), read(3)) else {
        return;
    };
    consume(1 + REQUEST_WORDS);
    let code = if !in_range(addr, len) {
        Code::BadRange
    } else {
        let source = ptr::without_provenance::<u64>(addr as usize);
        let destination = ptr::without_provenance_mut::<u64>(IMAGE_BASE);
        // SAFETY: the range owns the source memory and the bootloader owns the destination;
        // `Image::land` writes and hashes the same local value from each source word.
        Image {
            words: len as usize / 8,
            token,
            digest: FIRMWARE_DIGEST,
        }
        .land(
            |word| unsafe { ptr::read_volatile(source.add(word)) },
            |word, value| unsafe { ptr::write_volatile(destination.add(word), value) },
        )
    };
    complete(id, code);
    if code == Code::Booted {
        // SAFETY: `land` put an image opening with the magic at `IMAGE_BASE`; its entry follows.
        unsafe {
            asm!("dsb sy", "isb", "br {entry}", entry = in(reg) IMAGE_BASE + 8, options(noreturn));
        }
    }
}

/// Publishes one completion answering `id`, waiting for room.
fn complete(id: u32, code: Code) {
    let words = [header::pack(id, REPLY_WORDS as u8), code as u64, 0, 0];
    if let Ok(completion) = Producer::new(COMPLETION) {
        let _ = block_on(|| completion.write_all(&words));
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
