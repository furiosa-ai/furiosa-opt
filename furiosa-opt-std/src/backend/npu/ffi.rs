#[expect(dead_code, missing_docs, reason = "bindgen output; bindgen does not emit docs")]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::NpuDesc;
pub(crate) use bindings::*;

use std::sync::OnceLock;

use crate::context::Device;

pub(crate) fn rt() -> *const Runtime {
    struct Handle(*mut Runtime);
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    static RT: OnceLock<Handle> = OnceLock::new();
    RT.get_or_init(|| {
        let ver = unsafe { std::ffi::CStr::from_ptr(furiosa_version()) };
        log::info!("device-runtime: {}", ver.to_string_lossy());
        let Device { chip: chips, pe: pes } = *DEVICE.get_or_init(|| Device { chip: 1, pe: 8 });

        // Optimistic acquisition: `furiosa_runtime_init`'s exclusive open (null when a
        // chip is busy) is the atomic arbiter, so try each candidate set of `chips`
        // and take the first that acquires. No check-then-use, hence no TOCTOU race;
        // the chips are whatever the host exposes and need not be contiguous.
        let available = available_chips();
        for group in available.windows(chips as usize) {
            let npus: Vec<NpuDesc> = group
                .iter()
                .flat_map(|&chip| match pes {
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
            let ptr = unsafe { furiosa_runtime_init(npus.as_ptr(), npus.len()) };
            if !ptr.is_null() {
                return Handle(ptr);
            }
        }
        panic!("no NPU available: {} chip(s) present, all in use", available.len());
    })
    .0
}

/// Chip ids the host exposes, read from the `/dev/rngd/npu<N>mgmt` device nodes.
fn available_chips() -> Vec<u8> {
    let mut chips: Vec<u8> = std::fs::read_dir("/dev/rngd")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            name.to_str()?.strip_prefix("npu")?.strip_suffix("mgmt")?.parse().ok()
        })
        .collect();
    chips.sort_unstable();
    chips
}

static DEVICE: OnceLock<Device> = OnceLock::new();

/// Binds the `#[device(chip, pe)]` the process runs on, via `Context::acquire().bind(..)`.
/// First bind wins; a conflicting one panics.
pub(crate) fn bind_device(device: Device) {
    let cur = *DEVICE.get_or_init(|| device);
    assert_eq!(
        cur, device,
        "conflicting NPU device in one process: {cur:?} vs {device:?}"
    );
}
