//! Vector Engine examples and tests.
//!
//! 단순 커버리지를 위한 테스트들 모음입니다.
//!
//! This module provides examples for:
//! - `ve_elementwise_*`: VectorTensor pipeline operations (single input) - vector_init()으로 생성된 vector engine
//! - `ve_group_pair_*`: VectorTensorPair two-group operations (two inputs) - vector_init_unzip()으로 생성된 vector engine

use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2];
axes![
    A = 512,
    B = 256,
    I = 2,
    Q = 2,
    R = 4,
    R16 = 16,
    S = 15,
    T = 4,
    P = 8,
    W = 64
];

mod normal;
mod reduce;
mod zip;

pub use normal::*;
pub use reduce::*;
pub use zip::*;
