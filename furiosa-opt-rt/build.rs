//! Embeds the two vendored images every cluster runs.

use std::path::Path;

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=prebuilt");

    let bootloader = manifest.join("prebuilt/furiosa-opt-bootloader.signed");
    let firmware = manifest.join("prebuilt/furiosa-opt-firmware.bin");
    println!("cargo:rustc-env=FURIOSA_OPT_BOOTLOADER={}", bootloader.display());
    println!("cargo:rustc-env=FURIOSA_OPT_DEVICE_IMAGE={}", firmware.display());
}
