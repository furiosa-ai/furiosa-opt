# furiosa-visa-std Crate Context

Virtual ISA standard library. Handles tensor system and vector engine operations.

## Module Structure

```
furiosa-visa-std/src/
├── lib.rs              # Crate entry, prelude exports
├── tensor.rs           # Tensor trait definitions
├── raw_tensor.rs       # Low-level tensor implementation
├── memory_tensor.rs    # Memory-backed tensor
├── stream_tensor.rs    # Streaming tensor for VE operations
├── tensor_view.rs      # Borrowed tensor views
├── context.rs          # Execution context management
├── runtime.rs          # Runtime environment
├── scalar.rs           # Scalar type definitions
├── cast.rs             # Type casting utilities
├── array_vec.rs        # Fixed-size array vector
├── mapping_conversion.rs # Mapping conversions
└── vector_engine/      # Vector Engine subsystem
    ├── mod.rs
    ├── alu.rs          # ALU operations
    ├── branch.rs       # Branch unit configuration
    ├── config.rs       # VE configuration types
    ├── operand.rs      # VE operand types
    ├── vrf_array.rs    # VRF array management
    ├── vrf_list.rs     # VRF list utilities
    ├── op/             # Operation implementations
    ├── stage/          # Pipeline stage markers
    └── tensor/         # VE tensor types
        ├── vector_tensor.rs      # VectorTensor for pipeline stages
        └── binary_split_tensor.rs # VectorTensorPair for two-group ops
```

## Feature Flags

This crate uses nightly Rust features:
- `adt_const_params`: ADT const generics
- `inherent_associated_types`: Inherent associated types

## Lints

```rust
#![warn(missing_docs)]              // Documentation required for all public items
#![warn(missing_debug_implementations)]  // Debug implementation required
#![forbid(unused_must_use)]         // Must not ignore must_use return values
```

## Coding Conventions (furiosa-visa-std specific)

### Tensor Types
- `RawTensor`: Lowest level, direct memory manipulation
- `MemoryTensor`: Tensor bound to a specific memory region
- `StreamTensor`: Streaming tensor that interacts with vector engine

### Vector Engine
- `config.rs`: VE configuration type definitions (modify with caution)
- `branch.rs`: Branch unit configuration and apply_branch_config
- `operand.rs`: VE operand types (VeRhs, BranchOperand, VeOperand, IntoOperands)
- `op/`: Operation definitions and semantics
- `stage/`: Pipeline stage markers and transitions
- `tensor/`: VectorTensor and VectorTensorPair tensor types

### Public API
- All public items require `///` documentation comments
- Re-export main types through `prelude` module
- Re-export `device`, `m` macros from `furiosa_opt_macro`

## Constraints (Do NOT)

- Due to `#![forbid(unused_must_use)]`, ignoring Result/Option causes compile error
- `missing_docs` warning is enabled, so documentation is required for public items
- `clippy::type_complexity` is allowed, but prefer using type aliases when possible
