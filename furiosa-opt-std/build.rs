fn main() {
    // The `backend` cfg is npu-only: absence means the CPU backend, so nothing is injected.
    println!("cargo:rerun-if-env-changed=CARGO_CFG_BACKEND");
    let backend = std::env::var("CARGO_CFG_BACKEND").unwrap_or_default();

    let target = std::env::var("TARGET").unwrap();
    let manifest = env!("CARGO_MANIFEST_DIR");

    // cbindgen header is arch-independent; the single committed copy serves every target.
    let header = format!("{manifest}/vendor/x86_64-unknown-linux-gnu/device_runtime.h");

    println!("cargo:rerun-if-changed={header}");

    // Only the NPU backend links the device runtime; other backends never reference
    // it. Fail early if an npu build is missing the vendored archive.
    if backend == "npu" {
        let lib = format!("{manifest}/vendor/{target}/libdevice_runtime.a");
        assert!(
            std::path::Path::new(&lib).exists(),
            "backend=\"npu\" needs the vendored device runtime at {lib}; run `make furiosa-opt-vendor`"
        );
        println!("cargo:rerun-if-changed={lib}");
        println!("cargo:rustc-link-search=native={manifest}/vendor/{target}");
        println!("cargo:rustc-link-lib=static=device_runtime");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=gcc_s");
    }

    bindgen::Builder::default()
        .header(header)
        .allowlist_function("furiosa_.*")
        .allowlist_type("NpuDesc")
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(format!("{}/bindings.rs", std::env::var("OUT_DIR").unwrap()))
        .expect("failed to write bindings");
}
