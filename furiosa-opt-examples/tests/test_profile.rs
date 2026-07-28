//! Minimal on-device profiling example. A profile client installs a `tracing`
//! layer that observes the `span::npu` target; each decoded TUC span arrives as
//! an `info_span!` carrying its cycle window. This counts them and asserts the
//! profiled run produced spans. Gated to the `npu` backend, so it runs under:
//!
//!   TUC_PROFILE_LEVEL=info \
//!     cargo furiosa-opt --backend npu test -p furiosa-opt-examples --test test_profile
#![cfg(backend = "npu")]
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use furiosa_opt_examples::contract_element_types::{A, K8, R, i8_contract};
use furiosa_opt_std::prelude::*;
use tracing_subscriber::layer::{Context as LayerContext, Layer};
use tracing_subscriber::prelude::*;

type Chip = m![1];

/// Counts on-device profile spans (target `span::npu`) as they are emitted.
#[derive(Clone, Default)]
struct Counter(Arc<AtomicUsize>);

impl Counter {
    fn incr(&self) -> usize {
        self.0.fetch_add(1, Ordering::Relaxed)
    }

    fn read(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

impl<S: tracing::Subscriber> Layer<S> for Counter {
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, _id: &tracing::span::Id, _ctx: LayerContext<'_, S>) {
        if attrs.metadata().target() == "span::npu" {
            self.incr();
        }
    }
}

#[tokio::test]
async fn profile_i8_contract() {
    // Separate from the span assertion below: the runtime records nothing under
    // `info`, and that must not read as a regression in the profiled path.
    let level = std::env::var("TUC_PROFILE_LEVEL")
        .unwrap_or_default()
        .to_ascii_lowercase();
    assert!(
        matches!(level.as_str(), "info" | "debug" | "trace"),
        "TUC_PROFILE_LEVEL is {level:?}; the runtime only records spans at info or above",
    );

    let counter = Counter::default();
    tracing_subscriber::registry().with(counter.clone()).init();

    let mut ctx = Context::acquire();
    let input = HostTensor::<i8, m![A, K8]>::from_vec(vec![1; <m![A, K8]>::SIZE]);
    let trf = HostTensor::<i8, m![R, K8]>::from_vec(vec![1; <m![R, K8]>::SIZE]);
    let input_hbm = input.to_hbm::<Chip, m![A, K8]>(&mut ctx.pdma).await;
    let trf_hbm = trf.to_hbm::<Chip, m![R, K8]>(&mut ctx.pdma).await;

    let _ = launch(i8_contract, (&mut *ctx, &input_hbm, &trf_hbm)).await;

    // Spans are decoded off the launch hot path, so wait for the deferred
    // read-back before checking.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    assert!(counter.read() > 0, "expected on-device profile spans");
}
