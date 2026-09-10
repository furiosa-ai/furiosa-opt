//! TRF double-buffering schedules produced by rolled, pipelined, and fully unrolled loops.

use furiosa_opt_std::prelude::*;

mod rolled_kernel;
mod software_pipelined_kernel;
mod unrolled_kernel;

pub use rolled_kernel::rolled;
pub use software_pipelined_kernel::software_pipelined;
pub use unrolled_kernel::unrolled;

axes![Tok = 16, Red = 64, Out = 8, Group = 20, Pairs = 10];
