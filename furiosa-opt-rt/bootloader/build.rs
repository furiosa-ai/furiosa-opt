use sha2::{Digest, Sha256};

fn main() {
    println!("cargo:rerun-if-changed=bootloader.lds");
    println!("cargo:rerun-if-env-changed=FURIOSA_OPT_FIRMWARE");
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        return;
    }
    let firmware = std::path::PathBuf::from(
        std::env::var_os("FURIOSA_OPT_FIRMWARE")
            .expect("FURIOSA_OPT_FIRMWARE must name the firmware image bound to this bootloader"),
    );
    println!("cargo:rerun-if-changed={}", firmware.display());
    let digest = Sha256::digest(std::fs::read(&firmware).expect("failed to read FURIOSA_OPT_FIRMWARE"));
    std::fs::write(
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR")).join("firmware.sha256"),
        digest,
    )
    .expect("failed to write the firmware digest");
    println!(
        "cargo:rustc-link-arg=-T{}",
        std::path::Path::new("bootloader.lds")
            .canonicalize()
            .expect("bootloader.lds must sit beside build.rs")
            .display(),
    );
}
