//! Device-function execution runtime: [`launch`] dispatches a [`DeviceFn`] over its arguments,
//! [`DeviceSend`] marks the types that may cross to a device function, and [`TupleApply`] adapts
//! tuple arguments to positional calls. This layer sits above the [`crate::backend`] abstraction
//! and drives it.

use cfg_if::cfg_if;

// Only the cfg-selected backend below is named in the active build; the other is referenced
// solely by the inactive `cfg_if` branch.
#[allow(unused_imports)]
use crate::backend::{Cpu, Npu};

cfg_if! {
    if #[cfg(backend = "npu")] {
        /// Backend alias used when compiling for the NPU runtime.
        pub type CurrentBackend = Npu;
    } else {
        /// Backend alias used for the host-side buffer emulator, the default: a plain
        /// `cargo build` and a `--cfg backend="CPU"` build are the same build.
        pub type CurrentBackend = Cpu;
    }
}

/// Trait for applying a function to arguments.
///
/// Allows `launch(f, (a, b, c))` to call `f(a, b, c)` instead of `f((a, b, c))`. Single reference args can be
/// passed directly without tuple wrapper.
pub trait TupleApply<Args> {
    /// Return type of the function.
    type Output;
    /// Apply the function to the arguments.
    fn apply(self, args: Args) -> Self::Output;
}

impl<F, A, R> TupleApply<&mut A> for F
where
    F: FnOnce(&mut A) -> R,
{
    type Output = R;
    fn apply(self, a: &mut A) -> R {
        self(a)
    }
}

impl<F, A, R> TupleApply<&A> for F
where
    F: FnOnce(&A) -> R,
{
    type Output = R;
    fn apply(self, a: &A) -> R {
        self(a)
    }
}

macro_rules! impl_tuple_apply {
    ($($T:ident),+) => {
        #[expect(non_snake_case, reason = "type parameters A..Z used as destructuring variable names")]
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
impl_tuple_apply!(A, B, C, D, E, G);
impl_tuple_apply!(A, B, C, D, E, G, H);
impl_tuple_apply!(A, B, C, D, E, G, H, I);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P, Q);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P, Q, R);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P, Q, R, S);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V);
impl_tuple_apply!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);

/// Marker trait for types that can be sent to device functions.
///
/// # Implements DeviceSend
///
/// - Scalars: `bool`, `i8`-`i64`, `u8`-`u64`, `usize`, `isize`, `f32`, `f64`
/// - Device memory types: `HbmTensor`, `HbmTensorView`, `HbmTensorViewMut`
/// - Context types: `&mut Context`
/// - Tuples of DeviceSend types (for argument composition)
///
/// # Does NOT implement DeviceSend
///
/// - `HostTensor` - lives in host memory
/// - `Vec<T>`, `String`, etc. - general collections
///
/// User-defined structs opt in via `#[derive(DeviceSend)]`, which requires every
/// field to be `DeviceSend` (so a struct is `DeviceSend` iff all its fields are).
pub trait DeviceSend {}

impl DeviceSend for () {}
impl DeviceSend for bool {}
impl DeviceSend for i8 {}
impl DeviceSend for i16 {}
impl DeviceSend for i32 {}
impl DeviceSend for i64 {}
impl DeviceSend for isize {}
impl DeviceSend for u8 {}
impl DeviceSend for u16 {}
impl DeviceSend for u32 {}
impl DeviceSend for u64 {}
impl DeviceSend for usize {}
impl DeviceSend for f32 {}
impl DeviceSend for f64 {}

macro_rules! impl_device_send_tuple {
    ($($T:ident),+) => {
        impl<$($T: DeviceSend),+> DeviceSend for ($($T,)+) {}
    };
}

impl_device_send_tuple!(A);
impl_device_send_tuple!(A, B);
impl_device_send_tuple!(A, B, C);
impl_device_send_tuple!(A, B, C, D);
impl_device_send_tuple!(A, B, C, D, E);
impl_device_send_tuple!(A, B, C, D, E, F);
impl_device_send_tuple!(A, B, C, D, E, F, G);
impl_device_send_tuple!(A, B, C, D, E, F, G, H);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V);
impl_device_send_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);

impl<T> DeviceSend for std::marker::PhantomData<T> {}

/// Device function trait, generated by `#[device]` macro.
///
/// `cargo <subcommand>`: `execute()` runs the original function body on CPU. `cargo furiosa-opt <subcommand>`:
/// `execute()` loads the compiled EDF and runs on NPU.
pub trait DeviceFn<Args: DeviceSend> {
    /// Return type of the device function.
    type Output: DeviceSend;
    /// Execute the device function.
    fn execute(args: Args) -> impl std::future::Future<Output = Self::Output>;
}

/// Launches a device function. Takes `F` by value so callers can pass the snake_case const emitted by
/// `#[device]` (`launch(my_fn, args)`) rather than turbofishing the generated PascalCase unit struct
/// (`<MyFn as DeviceFn<_>>::execute(args)`). The value is discarded; only its type drives trait dispatch.
pub async fn launch<F, P>(_f: F, args: P) -> F::Output
where
    F: DeviceFn<P>,
    P: DeviceSend,
{
    F::execute(args).await
}
