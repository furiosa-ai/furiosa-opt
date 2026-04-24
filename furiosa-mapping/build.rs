fn main() {
    lalrpop::process_root().unwrap();

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let impl_target_dir = out_dir.join("furiosa-mapping-impl-target");

    let lib_name = "furiosa_mapping_impl";
    let lib = impl_target_dir.join(format!("release/lib{lib_name}.a"));

    #[allow(unused_assignments, unused_mut)]
    let mut is_public = true;

    if is_public {
        std::fs::create_dir_all(impl_target_dir.join("release")).expect("failed to create impl target dir");

        let url_output = std::process::Command::new("sh")
            .arg("-c")
            .arg("curl -fsSL https://api.github.com/repos/furiosa-ai/furiosa-opt/releases/latest | jq -r '.assets[] | select(.name | contains(\"libfuriosa_mapping_impl\")) | .browser_download_url'")
            .output()
            .expect("failed to query GitHub releases");
        assert!(url_output.status.success(), "failed to get release URL from GitHub");

        let url = String::from_utf8(url_output.stdout).unwrap();
        let url = url.trim();
        assert!(
            !url.is_empty(),
            "could not find libfuriosa_mapping_impl asset in latest release"
        );

        let status = std::process::Command::new("curl")
            .args(["-fsSL", url, "-o"])
            .arg(&lib)
            .status()
            .expect("failed to run curl");
        assert!(status.success(), "failed to download libfuriosa_mapping_impl.a");
    }

    println!(
        "cargo:rustc-link-search=native={}",
        impl_target_dir.join("release").display()
    );
    println!("cargo:rustc-link-lib=static={}", lib_name);
    println!("cargo:rerun-if-changed={}", lib.display());
}
