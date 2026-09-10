//! Device-function execution runtime.

use std::fmt::Debug;

use arrayvec::ArrayVec;
use cfg_if::cfg_if;
use furiosa_opt_rt::{Buffer, Function};

use furiosa_opt_macro::primitive;

use crate::Error;
use crate::backend::Backend;
use crate::context::{Dma, DmaContext, Tu, TuContext};

// Only the cfg-selected backend below is named in the active build; the other is referenced
// solely by the inactive `cfg_if` branch.
#[allow(unused_imports)]
use crate::backend::{Cpu, Npu};

cfg_if! {
    if #[cfg(backend = "npu")] {
        /// Backend selected for an NPU build.
        pub type CurrentBackend = Npu;
    } else {
        /// Backend selected for a host build.
        pub type CurrentBackend = Cpu;
    }
}

/// The chips a device function runs on, declared by `#[device(chip, pe)]`.
pub use furiosa_opt_rt::Topology;

/// Device context.
#[primitive(Device)]
pub struct Device<B: Backend = CurrentBackend> {
    /// Tensor unit for the main context.
    pub main: TuContext<{ Tu::Main }>,
    /// Tensor unit for the sub context.
    pub sub: TuContext<{ Tu::Sub }>,
    /// Tensor DMA context.
    pub tdma: DmaContext<{ Dma::Tensor }, B>,
    /// PCIe DMA context.
    pub pdma: DmaContext<{ Dma::Pcie }, B>,
    // The kernel translator reads `main`, `sub`, `tdma`, `pdma` as MIR fields 0..=3.
}

impl<B: Backend> Debug for Device<B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Device").finish_non_exhaustive()
    }
}

impl<B: Backend> DeviceSend for Device<B> {
    fn bind(&self, _: &mut Buffers) -> Result<(), Error> {
        Ok(())
    }
}

impl Device {
    /// Opens the chips `topology` names; a device function's topology is `function.topology()`.
    ///
    /// Each call opens chips of its own, so two devices of one topology hold different PEs.
    pub fn new(topology: Topology) -> Result<Self, Error> {
        Ok(Self::on(CurrentBackend::open(topology)?))
    }
}

impl<B: Backend> Device<B> {
    pub(crate) fn on(device: B::Device) -> Self {
        Self {
            main: TuContext::<{ Tu::Main }>::on(),
            sub: TuContext::<{ Tu::Sub }>::on(),
            tdma: DmaContext::<{ Dma::Tensor }, B>::on(device.clone()),
            pdma: DmaContext::<{ Dma::Pcie }, B>::on(device),
        }
    }

    /// Prepares a launch of `function` on this context, with `args` as the parameters after it:
    /// `device.launch(f, (&a, &b))` is `launch(f, (&mut device, &a, &b))`.
    pub fn launch<'c, F, Args>(&'c mut self, function: F, args: Args) -> Launch<F, Args::Output>
    where
        Args: Prepend<&'c mut Self>,
        Args::Output: DeviceSend,
        F: DeviceFn<Args::Output>,
    {
        launch(function, args.prepend(self))
    }
}

/// Applies a function to its unpacked arguments.
pub trait TupleApply<Args> {
    /// Return type of the function.
    type Output;
    /// Invokes the function with `args`.
    fn apply(self, args: Args) -> Self::Output;
}

impl<F, A, R> TupleApply<&mut A> for F
where
    F: FnOnce(&mut A) -> R,
{
    type Output = R;

    fn apply(self, arg: &mut A) -> R {
        self(arg)
    }
}

impl<F, A, R> TupleApply<&A> for F
where
    F: FnOnce(&A) -> R,
{
    type Output = R;

    fn apply(self, arg: &A) -> R {
        self(arg)
    }
}

macro_rules! impl_tuple_apply {
    ($($T:ident),+) => {
        #[expect(non_snake_case, reason = "type parameters are destructuring names")]
        impl<Func, $($T,)+ Ret> TupleApply<($($T,)+)> for Func
        where
            Func: FnOnce($($T,)+) -> Ret,
        {
            type Output = Ret;

            fn apply(self, ($($T,)+): ($($T,)+)) -> Ret {
                self($($T,)+)
            }
        }
    };
}

impl_tuple_apply!(A, B);
impl_tuple_apply!(A, B, C);
impl_tuple_apply!(A, B, C, D);
impl_tuple_apply!(A, B, C, D, E);
impl_tuple_apply!(A, B, C, D, E, F);
impl_tuple_apply!(A, B, C, D, E, F, G);
impl_tuple_apply!(A, B, C, D, E, F, G, H);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V);
impl_tuple_apply!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);

/// A device function's parameters after its leading `&mut Device`, as [`crate::Device::launch`] takes
/// them; `prepend` puts the context back in front, in the shape the function's [`DeviceFn`] names.
pub trait Prepend<Head> {
    /// The parameters with `Head` in front: a lone `Head` when there are none.
    type Output;
    /// `head`, then `self`'s elements.
    fn prepend(self, head: Head) -> Self::Output;
}

impl<Head> Prepend<Head> for () {
    type Output = Head;

    fn prepend(self, head: Head) -> Head {
        head
    }
}

macro_rules! impl_prepend {
    ($($T:ident),+) => {
        #[expect(non_snake_case, reason = "type parameters are destructuring names")]
        impl<Head, $($T,)+> Prepend<Head> for ($($T,)+) {
            type Output = (Head, $($T,)+);

            fn prepend(self, head: Head) -> Self::Output {
                let ($($T,)+) = self;
                (head, $($T,)+)
            }
        }
    };
}

impl_prepend!(A);
impl_prepend!(A, B);
impl_prepend!(A, B, C);
impl_prepend!(A, B, C, D);
impl_prepend!(A, B, C, D, E);
impl_prepend!(A, B, C, D, E, F);
impl_prepend!(A, B, C, D, E, F, G);
impl_prepend!(A, B, C, D, E, F, G, H);
impl_prepend!(A, B, C, D, E, F, G, H, I);
impl_prepend!(A, B, C, D, E, F, G, H, I, J);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V);
impl_prepend!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);

/// The device buffers one launch carries, in declaration order. Lives on the launching stack: a
/// launch frame bounds the count, so no launch allocates.
pub type Buffers = ArrayVec<Buffer, { Function::MAX_ARGS }>;

/// A value a device function receives: it contributes its device buffers to the launch.
///
/// # Implements DeviceSend
///
/// - Device memory types: `HbmTensor`, `HbmTensorView`, `HbmTensorViewMut`
/// - Device types: `&mut Device`
/// - Tuples of DeviceSend types (for argument composition)
///
/// # Does NOT implement DeviceSend
///
/// - `HostTensor` - lives in host memory
/// - Scalars - no NPU binding representation
/// - `Vec<T>`, `String`, etc. - general collections
///
/// User-defined structs opt in via `#[derive(DeviceSend)]`, which requires every
/// field to be `DeviceSend` (so a struct is `DeviceSend` iff all its fields are).
pub trait DeviceSend {
    /// Pushes this value's device buffers onto `buffers`, in declaration order. Fails on an HBM
    /// tensor the runtime never placed.
    fn bind(&self, buffers: &mut Buffers) -> Result<(), Error>;
}

/// Runs a function emitted by `#[device]`.
///
/// `cargo <subcommand>`: `execute()` runs the original function body on CPU. `cargo furiosa-opt <subcommand>`:
/// `execute()` loads the compiled registry entry and runs on NPU.
///
/// Device functions accept and return only values that can cross the device boundary.
///
/// ```compile_fail,E0277
/// use furiosa_opt_std::runtime::DeviceFn;
///
/// struct Invalid;
///
/// impl DeviceFn<i32> for Invalid {
///     type Output = ();
///
///     async fn execute(_: i32) -> Result<(), furiosa_opt_std::Error> {
///         Ok(())
///     }
/// }
/// ```
///
/// ```compile_fail,E0277
/// use furiosa_opt_std::runtime::DeviceFn;
///
/// struct Invalid;
///
/// impl DeviceFn<()> for Invalid {
///     type Output = String;
///
///     async fn execute(_: ()) -> Result<String, furiosa_opt_std::Error> {
///         Ok(String::new())
///     }
/// }
/// ```
///
/// ```compile_fail,E0277
/// use furiosa_opt_std::runtime::{DeviceFn, DeviceSend};
///
/// struct Args;
/// struct Output;
/// struct Invalid;
///
/// impl DeviceSend for Args {
///     fn bind(&self, _: &mut furiosa_opt_std::runtime::Buffers) -> Result<(), furiosa_opt_std::Error> {
///         Ok(())
///     }
/// }
///
/// impl DeviceSend for Output {
///     fn bind(&self, _: &mut furiosa_opt_std::runtime::Buffers) -> Result<(), furiosa_opt_std::Error> {
///         Ok(())
///     }
/// }
///
/// impl DeviceFn<Args> for Invalid {
///     type Output = Output;
///
///     async fn execute(_: Args) -> Result<Output, furiosa_opt_std::Error> {
///         Ok(Output)
///     }
/// }
/// ```
pub trait DeviceFn<Args: DeviceSend> {
    /// Return type of the function.
    type Output: crate::__private::DeviceOutput;
    /// Runs the function.
    fn execute(args: Args) -> impl std::future::Future<Output = Result<Self::Output, Error>>;
    /// Runs the function into caller-owned output storage.
    fn execute_into(args: Args, output: &mut Self::Output) -> impl std::future::Future<Output = Result<(), Error>> {
        async move {
            *output = Self::execute(args).await?;
            Ok(())
        }
    }
}

#[must_use = "launches do nothing unless awaited"]
#[derive(Debug)]
/// A prepared device-function launch.
pub struct Launch<F, Args> {
    function: F,
    args: Args,
}

impl<F, Args> Launch<F, Args>
where
    F: DeviceFn<Args>,
    Args: DeviceSend,
{
    /// Binds the launch to caller-owned output storage.
    pub fn output(self, output: &mut F::Output) -> LaunchInto<'_, F, Args> {
        LaunchInto {
            function: self.function,
            args: self.args,
            output,
        }
    }
}

impl<F, Args> std::future::IntoFuture for Launch<F, Args>
where
    F: DeviceFn<Args>,
    Args: DeviceSend,
{
    type Output = Result<F::Output, Error>;
    type IntoFuture = impl std::future::Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        async move {
            let _ = self.function;
            F::execute(self.args).await
        }
    }
}

#[must_use = "launches do nothing unless awaited"]
#[derive(Debug)]
/// A launch with caller-owned output storage.
pub struct LaunchInto<'output, F, Args>
where
    F: DeviceFn<Args>,
    Args: DeviceSend,
{
    function: F,
    args: Args,
    output: &'output mut F::Output,
}

impl<F, Args> std::future::IntoFuture for LaunchInto<'_, F, Args>
where
    F: DeviceFn<Args>,
    Args: DeviceSend,
{
    type Output = Result<(), Error>;
    type IntoFuture = impl std::future::Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        async move {
            let _ = self.function;
            F::execute_into(self.args, self.output).await
        }
    }
}

/// Prepares a device-function launch.
pub fn launch<F, Args>(function: F, args: Args) -> Launch<F, Args>
where
    F: DeviceFn<Args>,
    Args: DeviceSend,
{
    Launch { function, args }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::npu::Function;
    use crate::{Device, Topology};

    struct Manual;

    struct Value(i32);

    impl DeviceSend for Value {
        fn bind(&self, _: &mut Buffers) -> Result<(), Error> {
            Ok(())
        }
    }

    impl crate::__private::DeviceOutput for Value {
        fn alloc_into(_: &Function, _: &mut Buffers) -> Result<(), Error> {
            Ok(())
        }

        fn take(_: &mut impl Iterator<Item = Buffer>) -> Self {
            Self(0)
        }
    }

    impl DeviceFn<Value> for Manual {
        type Output = Value;

        fn execute(value: Value) -> impl std::future::Future<Output = Result<Value, Error>> {
            std::future::ready(Ok(Value(value.0 + 1)))
        }
    }

    impl DeviceFn<(&mut Device, Value)> for Manual {
        type Output = Value;

        fn execute((_, value): (&mut Device, Value)) -> impl std::future::Future<Output = Result<Value, Error>> {
            std::future::ready(Ok(Value(value.0 + 10)))
        }
    }

    #[tokio::test]
    async fn output_launch_stores_result() {
        let mut output = Value(0);

        launch(Manual, Value(1)).output(&mut output).await.expect("launch");

        assert_eq!(output.0, 2);
    }

    #[tokio::test]
    async fn a_context_launches_with_itself_in_front() {
        let mut context = Device::new(Topology { chips: 1, pes: 8 }).expect("context");

        let output = context.launch(Manual, (Value(1),)).await.expect("launch");

        assert_eq!(output.0, 11);
    }

    use crate::context::{Dma, DmaContext};
    #[cfg(not(backend = "npu"))]
    use crate::prelude::*;

    #[test]
    fn defaults_to_current_backend() {
        let topology = Topology { chips: 1, pes: 8 };
        let _: Device = Device::new(topology).unwrap();
        let context: Device = Device::new(topology).unwrap();
        let _: &DmaContext<{ Dma::Pcie }> = &context.pdma;
    }

    #[test]
    fn kernel_visible_fields_lead() {
        let (_, body) = include_str!("runtime.rs").split_once("pub struct Device<").unwrap();
        let (body, _) = body.split_once("\n}").unwrap();
        let fields: Vec<_> = body
            .lines()
            .filter_map(|line| line.trim().strip_prefix("pub ")?.split_once(':'))
            .map(|(name, _)| name)
            .collect();
        assert_eq!(fields, ["main", "sub", "tdma", "pdma"]);
    }

    #[tokio::test]
    #[cfg(not(backend = "npu"))]
    async fn pcie_dma_round_trips() {
        axes![A = 4];

        let mut context = Device::new(Topology { chips: 1, pes: 8 }).expect("CPU context");
        let values = vec![3, 1, 4, 1];

        let hbm = HostTensor::<i32, m![A]>::from_vec(values.clone())
            .to_hbm::<m![1], m![A]>(&mut context.pdma)
            .await
            .unwrap();
        let host = hbm.to_host::<m![A]>(&mut context.pdma).await.unwrap();

        assert_eq!(host.into_vec(), values);
    }
}
