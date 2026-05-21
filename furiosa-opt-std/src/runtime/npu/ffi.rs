#[expect(
    dead_code,
    missing_docs,
    unsafe_op_in_unsafe_fn,
    reason = "bindgen output; bindgen does not emit docs or mark ffi wrappers as safe"
)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::NpuDesc;
pub(crate) use bindings::*;

use std::str::FromStr;
use std::sync::{LazyLock, OnceLock};

pub(crate) fn lib() -> &'static DeviceRuntime {
    static LIB: LazyLock<DeviceRuntime> = LazyLock::new(|| {
        let lib = unsafe { DeviceRuntime::new(env!("DEVICE_RUNTIME_SO")) }
            .unwrap_or_else(|e| panic!("failed to load libdevice_runtime.so: {e}"));
        let ver = unsafe { std::ffi::CStr::from_ptr(lib.furiosa_version()) };
        log::info!("device-runtime: {}", ver.to_string_lossy());
        lib
    });
    &LIB
}

pub(crate) fn rt() -> *const Runtime {
    struct Handle(*mut Runtime);
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    static RT: OnceLock<Handle> = OnceLock::new();
    RT.get_or_init(|| {
        let NpuDescList(npus) = std::env::var("FURIOSA_OPT_NPUS")
            .unwrap_or_default()
            .parse()
            .unwrap_or_else(|e| panic!("FURIOSA_OPT_NPUS: {e}"));
        let ptr = unsafe { lib().furiosa_runtime_init(npus.as_ptr(), npus.len()) };
        assert!(
            !ptr.is_null(),
            "failed to acquire NPU; set `FURIOSA_OPT_NPUS=<chip>[,<chip>...]` to use different chips",
        );
        Handle(ptr)
    })
    .0
}

/// Comma-separated chip IDs parsed from `FURIOSA_OPT_NPUS` (empty string defaults to chip 0).
///
/// Each chip binds both RNGD half-clusters — the only supported topology is logical `pe0-7` per chip
/// (a 2×4-PE fusion via file handles `/dev/rngd{N}/pe{0-3,4-7}`); narrower fusions like `pe0-3` or
/// `pe0-1` fail at runtime init with "Invalid device ID".
struct NpuDescList(Vec<NpuDesc>);

impl FromStr for NpuDescList {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tokens: Vec<&str> = s.split(',').map(str::trim).filter(|t| !t.is_empty()).collect();
        let tokens: &[&str] = if tokens.is_empty() { &["0"] } else { &tokens };
        let npus = tokens
            .iter()
            .map(|t| t.parse::<u8>().map_err(|e| format!("invalid chip `{t}`: {e}")))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flat_map(|chip| {
                [
                    NpuDesc {
                        chip,
                        pe_start: 0,
                        pe_end: 3,
                    },
                    NpuDesc {
                        chip,
                        pe_start: 4,
                        pe_end: 7,
                    },
                ]
            })
            .collect();
        Ok(Self(npus))
    }
}
