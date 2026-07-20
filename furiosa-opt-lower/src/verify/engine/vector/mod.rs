//! Vector-engine verifications, mirroring `furiosa-opt-std/src/engine/vector`.

mod tensor;

pub use tensor::{
    VectorError, config_reduce_label, config_vector_narrow_split, config_vector_narrow_trim,
    config_vector_widen_concat, config_vector_widen_pad,
};
