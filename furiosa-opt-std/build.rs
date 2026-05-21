fn main() {
    // Default `backend` cfg to "simulation" when no caller (RUSTFLAGS / .cargo/config) set one.
    // Cargo populates CARGO_CFG_BACKEND for build scripts when the cfg is set externally; an empty
    // or absent value means nothing was passed in.
    println!("cargo:rerun-if-env-changed=CARGO_CFG_BACKEND");
    if std::env::var("CARGO_CFG_BACKEND").unwrap_or_default().is_empty() {
        println!("cargo:rustc-cfg=backend=\"simulation\"");
    }

    let target = std::env::var("TARGET").unwrap();
    let dir = format!("{}/vendor/{target}", env!("CARGO_MANIFEST_DIR"));
    let header = format!("{dir}/device_runtime.h");
    let so = format!("{dir}/libdevice_runtime.so");

    println!("cargo:rerun-if-changed={header}");
    println!("cargo:rerun-if-changed={so}");
    println!("cargo:rustc-env=DEVICE_RUNTIME_SO={so}");

    bindgen::Builder::default()
        .header(header)
        .dynamic_library_name("DeviceRuntime")
        .dynamic_link_require_all(true)
        .allowlist_function("furiosa_.*")
        .allowlist_type("NpuDesc")
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(format!("{}/bindings.rs", std::env::var("OUT_DIR").unwrap()))
        .expect("failed to write bindings");
}
