//! Matrix multiplication implementations for various sizes.

pub mod matmul_16384;
pub mod matmul_4096;
pub mod matmul_chip_reduce;
pub mod matmul_cluster_reduce;
pub mod matmul_split_reduce;
pub mod matmul_split_reduce2;
pub mod matmul_wo_broadcast;

pub use matmul_4096::matmul_4096;
pub use matmul_16384::matmul_16384;
pub use matmul_chip_reduce::matmul_chip_reduce;
pub use matmul_cluster_reduce::matmul_cluster_reduce;
pub use matmul_split_reduce::matmul_with_split_reduce;
pub use matmul_split_reduce2::matmul_with_split_reduce2;
pub use matmul_wo_broadcast::matmul_wo_broadcast;
