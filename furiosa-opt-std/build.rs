use std::path::PathBuf;

const SIGNED_DEVICE_RUNTIME_ENV: &str = "FURIOSA_OPT_SIGNED_DEVICE_RUNTIME";

fn main() {
    // The `backend` cfg is npu-only: absence means the CPU backend, so nothing is injected.
    println!("cargo:rerun-if-env-changed=CARGO_CFG_BACKEND");
    let backend = std::env::var("CARGO_CFG_BACKEND").unwrap_or_default();

    let target = std::env::var("TARGET").unwrap();
    let manifest = env!("CARGO_MANIFEST_DIR");

    // cbindgen header is arch-independent; the single committed copy serves every target.
    let header = format!("{manifest}/vendor/x86_64-unknown-linux-gnu/device_runtime.h");

    println!("cargo:rerun-if-changed={header}");

    // Only the NPU backend links the device runtime; other backends never reference it.
    // A caller-provided signed archive takes precedence over the vendored development archive.
    if backend == "npu" {
        println!("cargo:rerun-if-env-changed={SIGNED_DEVICE_RUNTIME_ENV}");

        let lib = std::env::var_os(SIGNED_DEVICE_RUNTIME_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let lib = PathBuf::from(format!("{manifest}/vendor/{target}/libdevice_runtime.a"));
                assert!(
                    lib.is_file(),
                    "backend=\"npu\" needs the vendored device runtime at {}; run `make furiosa-opt-vendor`",
                    lib.display(),
                );
                lib
            });
        let lib_dir = lib.parent().expect("device runtime path must have a parent");
        println!("cargo:rerun-if-changed={}", lib.display());
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
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
