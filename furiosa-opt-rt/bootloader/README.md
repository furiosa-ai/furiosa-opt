# furiosa-opt-bootloader

The signed bootloader: what production chips insist on, kept to what the hardware makes obvious.
It initializes the core, then answers one request written in the kernel driver's queue entry format,
`[header][addr][len][token]`: authenticate the firmware image at `addr` in device memory, copy it
into the scratchpad and jump to it. The contract lives in `furiosa_opt_abi::bootloader`.

Its signed bytes contain the SHA-256 digest of one firmware image. A firmware change therefore
requires rebuilding and signing the bootloader even though the images remain separate.

## Build

```sh
make furiosa-opt-vendor SIGN=dev
```

The vendor target builds the firmware first and supplies it to the bootloader build as
`FURIOSA_OPT_FIRMWARE`.

`make furiosa-opt-bootloader-test` exercises the copy-and-authenticate path on the host, including
rejection of a changed firmware word and replacement of those rejected bytes by a valid retry.

## Signing

The device's secure boot accepts a bootloader only with a signature from a key it trusts, so the runtime embeds `furiosa-opt-bootloader.signed`, never the bare `bootloader.bin`.
The public distribution carries that signed image but not the signer, its certificates or the
release-key configuration.
