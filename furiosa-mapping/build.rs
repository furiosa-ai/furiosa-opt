fn main() {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let impl_target_dir = out_dir.join("furiosa-mapping-impl-target");

    let lib_name = "furiosa_mapping_impl";
    // Forward TARGET to the inner cargo (private) and the download URL
    // (public) so the .a matches the consumer triple under cross-compile.
    let target = std::env::var("TARGET")
        .expect("TARGET is always set by cargo for build scripts; if you're invoking build.rs by hand, set TARGET to the target triple");
    // Inner cargo's --target puts output under <target-dir>/<triple>/<profile>/.
    let release_dir = impl_target_dir.join(&target).join("release");
    let lib = release_dir.join(format!("lib{lib_name}.a"));

    #[allow(unused_assignments, unused_mut)]
    let mut is_public = true;

    std::fs::create_dir_all(&release_dir).expect("failed to create impl target dir");
    if let Ok(prebuilt) = std::env::var("FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT") {
        std::fs::copy(&prebuilt, &lib).expect("failed to copy FURIOSA_MAPPING_IMPL_LOCAL_PREBUILT");
    } else if is_public {
        // Public (crates.io) builds fetch a prebuilt
        // libfuriosa_mapping_impl.a from the furiosa-ai/furiosa-opt
        // release matching this crate's version, verify SHA256SUMS, and
        // link. SHA256SUMS catches corruption but is not provenance.
        // Set FURIOSA_MAPPING_IMPL_VERIFY_ATTESTATION=1 to additionally
        // check the sigstore attestation via `gh attestation verify`.

        // Pin the URL to this crate's version. Override with
        // FURIOSA_MAPPING_IMPL_VERSION (e.g. "v0.1.5") for RC testing or
        // to pin past a yanked release.
        let version = std::env::var("FURIOSA_MAPPING_IMPL_VERSION")
            .unwrap_or_else(|_| format!("v{}", std::env::var("CARGO_PKG_VERSION").unwrap()));
        let asset_name = format!("lib{lib_name}-{version}-{target}.a");
        let licenses_name = format!("lib{lib_name}-{version}-{target}.LICENSES.txt");
        let base = format!("https://github.com/furiosa-ai/furiosa-opt/releases/download/{version}");
        let asset_url = format!("{base}/{asset_name}");
        let licenses_url = format!("{base}/{licenses_name}");
        let sums_url = format!("{base}/SHA256SUMS");

        let sums_path = impl_target_dir.join("SHA256SUMS");
        // Apache-2.0 §4(d): land LICENSES.txt at $OUT_DIR/<licenses_name>
        // (cargo's documented build-script output location) so a
        // redistributor can find it.
        let licenses_path = out_dir.join(&licenses_name);
        run_curl(&sums_url, &sums_path);

        // Opt-in provenance check. Verifies the SHA256SUMS attestation
        // against sigstore Rekor; the per-file SHA256 checks below then
        // transitively trust the entries inside. Default off because it
        // requires gh ≥ 2.49.0 on PATH and a network call to Rekor.
        if std::env::var("FURIOSA_MAPPING_IMPL_VERIFY_ATTESTATION").as_deref() == Ok("1") {
            verify_attestation(&sums_path);
        }

        run_curl(&asset_url, &lib);
        run_curl(&licenses_url, &licenses_path);

        verify_sha256(&lib, &sums_path, &asset_name);
        verify_sha256(&licenses_path, &sums_path, &licenses_name);
    }

    println!("cargo:rustc-link-search=native={}", release_dir.display());
    println!("cargo:rustc-link-lib=static={}", lib_name);
    println!("cargo:rerun-if-changed=build.rs");
    // Emit unconditionally so the rerun policy matches across private
    // and public trees, avoiding stale-cache surprises on tree switch.
    println!("cargo:rerun-if-env-changed=FURIOSA_MAPPING_IMPL_VERSION");
}

#[allow(dead_code)]
fn verify_attestation(sums: &std::path::Path) {
    let status = std::process::Command::new("gh")
        .args(["attestation", "verify", "--repo", "furiosa-ai/furiosa-opt"])
        .arg(sums)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "FURIOSA_MAPPING_IMPL_VERIFY_ATTESTATION=1 requires gh \
                 (>= 2.49.0) on PATH; failed to invoke `gh`: {e}"
            )
        });
    assert!(
        status.success(),
        "gh attestation verify failed for {}; refusing to link without \
         sigstore-attested provenance.",
        sums.display()
    );
}

#[allow(dead_code)]
fn run_curl(url: &str, dst: &std::path::Path) {
    // GitHub releases hit transient 503s during region failover; retry.
    // --max-time 600 caps total wall time per call; without it
    // --retry-max-time only bounds retry delays and a 1 KiB/s transfer
    // hangs forever.
    let status = std::process::Command::new("curl")
        .args([
            "-fsSL",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--retry-max-time",
            "60",
            "--connect-timeout",
            "15",
            "--max-time",
            "600",
            url,
            "-o",
        ])
        .arg(dst)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke curl for {url}: {e}"));
    assert!(status.success(), "curl failed for {url}");
}

#[allow(dead_code)]
fn verify_sha256(file: &std::path::Path, sums_file: &std::path::Path, asset_name: &str) {
    // Resolve the expected hash explicitly so absent or malformed
    // entries fail loud (`sha256sum -c --ignore-missing` skips them).
    let sums =
        std::fs::read_to_string(sums_file).unwrap_or_else(|e| panic!("failed to read {}: {e}", sums_file.display()));
    let expected = sums
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            let hash = parts.next()?;
            let name = parts.next()?.trim_start_matches('*');
            if name == asset_name {
                Some(hash.to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("no SHA256 entry for {asset_name} in {}", sums_file.display()));

    // Host sha256sum required; abort rather than skip verification.
    let output = std::process::Command::new("sha256sum").arg(file).output().expect(
        "sha256sum is required to verify the downloaded \
             libfuriosa_mapping_impl archive; install coreutils and retry",
    );
    assert!(
        output.status.success(),
        "sha256sum exited {} for {}: stderr={:?}",
        output
            .status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "<signaled>".into()),
        file.display(),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let computed = stdout
        .split_whitespace()
        .next()
        .unwrap_or_else(|| {
            panic!(
                "sha256sum produced empty output for {}; stdout was: {:?}",
                file.display(),
                stdout
            )
        })
        .to_string();
    assert_eq!(
        computed, expected,
        "SHA256 mismatch for {asset_name}: expected {expected}, got {computed}. \
         The downloaded archive does not match its declared checksum; refusing to link."
    );
}
