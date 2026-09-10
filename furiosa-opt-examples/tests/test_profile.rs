//! Minimal on-device profiling example. A profile client installs a `tracing`
//! layer that observes the `span::npu` target; each decoded TUC span arrives as
//! an `info_span!` carrying its cycle window. This counts them and asserts the
//! profiled run produced spans. Gated to the `npu` backend, so it runs under:
//!
//!   FURIOSA_OPT_PROFILE=info \
//!     cargo furiosa-opt test -p furiosa-opt-examples --test test_profile
#![cfg(backend = "npu")]
use std::sync::{Arc, Mutex};

use furiosa_opt_examples::contract_element_types::{A, K8, R, i8_contract};
use furiosa_opt_std::prelude::*;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context as LayerContext, Layer};
use tracing_subscriber::prelude::*;

type Chip = m![1];

/// Counts on-device profile spans (target `span::npu`) as they are emitted.
#[derive(Clone, Default)]
struct Counter(Arc<Mutex<Vec<Fields>>>);

#[derive(Default)]
struct Fields {
    category: bool,
    name: bool,
    begin: bool,
    end: bool,
}

impl Counter {
    fn read(&self) -> usize {
        self.0.lock().expect("profile spans").len()
    }

    fn verifies(&self) -> bool {
        self.0
            .lock()
            .expect("profile spans")
            .iter()
            .all(|fields| fields.category && fields.name && fields.begin && fields.end)
    }
}

impl Visit for Fields {
    fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}

    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "cat" => self.category = value == "NPU",
            "name" => self.name = !value.is_empty(),
            _ => {}
        }
    }

    fn record_u64(&mut self, field: &Field, _: u64) {
        match field.name() {
            "begin_cycle" => self.begin = true,
            "end_cycle" => self.end = true,
            _ => {}
        }
    }
}

impl<S: tracing::Subscriber> Layer<S> for Counter {
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, _id: &tracing::span::Id, _ctx: LayerContext<'_, S>) {
        if attrs.metadata().target() == "span::npu" {
            assert_eq!(attrs.metadata().name(), "NPU");
            let mut fields = Fields::default();
            attrs.record(&mut fields);
            self.0.lock().expect("profile spans").push(fields);
        }
    }
}

#[tokio::test]
async fn profile_i8_contract() {
    // Separate from the span assertion below: the runtime records nothing under
    // `info`, and that must not read as a regression in the profiled path.
    let level = std::env::var("FURIOSA_OPT_PROFILE")
        .unwrap_or_default()
        .to_ascii_lowercase();
    assert!(
        matches!(level.as_str(), "info" | "debug" | "trace"),
        "FURIOSA_OPT_PROFILE is {level:?}; the runtime only records spans at info or above",
    );

    let counter = Counter::default();
    tracing_subscriber::registry().with(counter.clone()).init();

    let mut device = Device::new(i8_contract.topology()).unwrap();
    let input = HostTensor::<i8, m![A, K8]>::from_vec(vec![1; <m![A, K8]>::SIZE]);
    let trf = HostTensor::<i8, m![R, K8]>::from_vec(vec![1; <m![R, K8]>::SIZE]);
    let input_hbm = input.to_hbm::<Chip, m![A, K8]>(&mut device.pdma).await.unwrap();
    let trf_hbm = trf.to_hbm::<Chip, m![R, K8]>(&mut device.pdma).await.unwrap();

    let _ = launch(i8_contract, (&mut device, &input_hbm, &trf_hbm)).await.unwrap();

    assert!(counter.read() > 0, "expected on-device profile spans");
    assert!(counter.verifies());
}
