fn main() {
    let target = std::env::var("TARGET").unwrap();
    let dir = format!("{}/vendor/{target}", env!("CARGO_MANIFEST_DIR"));
    let header = format!("{dir}/device_runtime.h");

    println!("cargo:rustc-env=DEVICE_RUNTIME_SO={dir}/libdevice_runtime.so");

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
