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
        let (chips, pes) = *DEVICE.get_or_init(|| (1, 8));
        let npus: Vec<NpuDesc> = (0..chips)
            .flat_map(|chip| match pes {
                8 => vec![
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
                ],
                n => vec![NpuDesc {
                    chip,
                    pe_start: 0,
                    pe_end: n - 1,
                }],
            })
            .collect();
        let ptr = unsafe { lib().furiosa_runtime_init(npus.as_ptr(), npus.len()) };
        assert!(!ptr.is_null(), "failed to acquire NPU");
        Handle(ptr)
    })
    .0
}

static DEVICE: OnceLock<(u8, u8)> = OnceLock::new();

/// Sets the `#[device(chip, pe)]` the process runs on. First call wins; a conflicting one panics.
pub fn set_device(chip: usize, pe: usize) {
    let dev = (chip as u8, pe as u8);
    let cur = *DEVICE.get_or_init(|| dev);
    assert_eq!(cur, dev, "conflicting NPU device in one process: {cur:?} vs {dev:?}");
}
