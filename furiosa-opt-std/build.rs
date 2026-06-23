fn main() {
    // Default `backend` cfg to "simulation" when no caller (RUSTFLAGS / .cargo/config) set one.
    // Cargo populates CARGO_CFG_BACKEND for build scripts when the cfg is set externally; an empty
    // or absent value means nothing was passed in.
    println!("cargo:rerun-if-env-changed=CARGO_CFG_BACKEND");
    if std::env::var("CARGO_CFG_BACKEND").unwrap_or_default().is_empty() {
        println!("cargo:rustc-cfg=backend=\"simulation\"");
    }

    let target = std::env::var("TARGET").unwrap();
    let manifest = env!("CARGO_MANIFEST_DIR");

    // cbindgen header is arch-independent; the single committed copy serves every target.
    let header = format!("{manifest}/vendor/x86_64-unknown-linux-gnu/device_runtime.h");

    // dlopened at runtime only by the NPU backend (`#[cfg(backend = "npu")]`);
    // simulation/typecheck never open it, so the path need not exist for them.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    // Keep this OS→extension mapping in sync with the `furiosa-opt-vendor` case
    // in the root Makefile, which writes the file this resolves.
    let lib_ext = match target_os.as_str() {
        "linux" => "so",
        "macos" => "dylib",
        "windows" => "dll",
        other => panic!("unsupported target OS for device-runtime: {other}"),
    };
    let so = format!("{manifest}/vendor/{target}/libdevice_runtime.{lib_ext}");

    println!("cargo:rerun-if-changed={header}");
    // Only watch the shared library when it actually exists — on hosts that
    // vendor no `device-runtime` (simulation/typecheck on non-x86), watching a
    // missing path would force a build-script re-run on every build.
    if std::path::Path::new(&so).exists() {
        println!("cargo:rerun-if-changed={so}");
    }
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
