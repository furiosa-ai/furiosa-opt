#[expect(dead_code, missing_docs, reason = "bindgen output; bindgen does not emit docs")]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::NpuDesc;
pub(crate) use bindings::*;

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::context::Device;

/// `tracing` target and category for on-device profile spans, mirroring
/// npu-executor's `TRACING_TARGET_NPU`/`TRACING_CATEGORY_NPU` so the shared
/// furiosa-telemetry chrome exporter consumes VISA kernel spans identically.
const TRACING_TARGET_NPU: &str = "span::npu";
const TRACING_CATEGORY_NPU: &str = "NPU";

/// Run a loaded kernel over its I/O buffers, arming on-device profiling only when
/// a subscriber listens on the `span::npu` target. Returns the runtime status
/// (0 on success). The enable check is the sole hot-path cost when profiling is
/// off; read-back and decode are deferred by the runtime, so the launch itself
/// never blocks.
pub(super) fn run(kernel: *mut Kernel, inputs: &[*const NpuBuffer], outputs: &[*const NpuBuffer]) -> i32 {
    if !tracing::enabled!(target: TRACING_TARGET_NPU, tracing::Level::INFO) {
        // SAFETY: the kernel handle and pointer arrays stay live across this synchronous call.
        return unsafe {
            furiosa_kernel_run(
                kernel,
                rt(),
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
            )
        };
    }

    // `sink` emits each decoded span as an `info_span!` on `span::npu` (name +
    // cycle window), matching npu-executor's schema; `done` is a noop since the
    // ctx is `null`. The runtime calls them off the launch hot path during
    // deferred read-back. They are `extern "C" fn` (not closures) because the C
    // ABI callback slots require the C ABI, which Rust closures cannot carry.
    unsafe extern "C" fn sink(_ctx: *mut c_void, name: Str, begin: u64, end: u64) {
        // SAFETY: the runtime passes a valid name pointer and length for this call.
        let bytes = unsafe { std::slice::from_raw_parts(name.ptr.cast::<u8>(), name.len) };
        tracing::info_span!(
            target: TRACING_TARGET_NPU,
            "NPU",
            cat = TRACING_CATEGORY_NPU,
            name = String::from_utf8_lossy(bytes).as_ref(),
            begin_cycle = begin,
            end_cycle = end,
        );
    }
    unsafe extern "C" fn done(_ctx: *mut c_void) {}

    // SAFETY: the handle and pointer arrays stay live across the launch; the sink/done callbacks
    // are `'static` and the ctx is `null`, so nothing must outlive this call.
    unsafe {
        furiosa_profiled_run(
            rt(),
            kernel,
            inputs.as_ptr(),
            inputs.len(),
            outputs.as_ptr(),
            outputs.len(),
            Some(sink),
            Some(done),
            std::ptr::null_mut(),
        )
    }
}

pub(crate) fn rt() -> *mut Runtime {
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
