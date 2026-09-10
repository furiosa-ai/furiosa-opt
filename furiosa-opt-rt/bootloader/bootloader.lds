/* Boot image layout: the first 64 KiB of the scratchpad, below the resident image. */

MEMORY
{
  RAM (rwx) : org = 0x00000000, l = 0x10000
}

ENTRY(_reset_entry)

SECTIONS
{
  /DISCARD/ : {
    *(.eh_frame_hdr)
    *(.eh_frame)
    *(.note*)
  }

  .text : {
    *(.text.reset_vector)
    *(.text)
    *(.text.*)
  } > RAM

  .rodata : {
    *(.rodata*)
  } > RAM

  .data : {
    *(.data)
    *(.data.*)
  } > RAM

  .bss ALIGN(16) : {
    PROVIDE(__bss_start__ = ABSOLUTE(.));
    *(.bss*)
    *(COMMON)
    . = ALIGN(8);
    PROVIDE(__bss_end__ = ABSOLUTE(.));
  } > RAM

  .stack ALIGN(16) (NOLOAD) : {
    . += 0x2000;
    __boot_stack_end = ABSOLUTE(.);
  } > RAM
}
