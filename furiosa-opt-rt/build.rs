//! Embeds the default or caller-provided images every cluster runs.

use std::path::Path;

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=prebuilt");

    for (variable, filename, output) in [
        (
            "FURIOSA_OPT_SIGNED_BOOTLOADER_PATH",
            "furiosa-opt-bootloader.signed",
            "FURIOSA_OPT_BOOTLOADER",
        ),
        (
            "FURIOSA_OPT_SIGNED_FIRMWARE_PATH",
            "furiosa-opt-firmware.bin",
            "FURIOSA_OPT_DEVICE_IMAGE",
        ),
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
        let image = std::env::var_os(variable)
            .map(|path| manifest.join(path))
            .unwrap_or_else(|| manifest.join("prebuilt").join(filename));
        println!("cargo:rerun-if-changed={}", image.display());
        println!("cargo:rustc-env={output}={}", image.display());
    }
}
