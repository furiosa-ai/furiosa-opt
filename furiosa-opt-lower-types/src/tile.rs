//! Typed failure reasons for pad and tile view verification.

use abi_stable::StableAbi;
use abi_stable::std_types::RBox;
use furiosa_mapping_types::Mapping;

/// Why a pad view is not a valid re-declaration of its buffer.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PadError {
    /// The requested padded view is narrower than the view it pads.
    #[error(
        "pad: the padded view cannot be narrower than the view it pads, but {expected} is \
         {extent} cells against {element}'s {live}"
    )]
    NarrowerView {
        /// The requested padded view.
        expected: Mapping,
        /// The requested extent in cells.
        extent: usize,
        /// The view being padded.
        element: Mapping,
        /// The current extent in cells.
        live: usize,
    },
    /// Padding the input to the requested extent does not produce the requested view.
    #[error(
        "pad: the requested view type is not the padded view. Padding {element} to {extent} \
         produces {padded}, but the requested view type is {expected}"
    )]
    UnexpectedView {
        /// The view being padded.
        element: RBox<Mapping>,
        /// The requested extent in cells.
        extent: usize,
        /// The view produced by padding.
        padded: RBox<Mapping>,
        /// The requested padded view.
        expected: RBox<Mapping>,
    },
}

/// Why a tile view cannot be split from its buffer.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TileError {
    /// The index expression does not split the element view as requested.
    #[error("tile: failed to split by the index expression")]
    Split,
    /// The split view differs from the requested view.
    #[error(
        "tile: the view type after the split does not match the requested view. \
         The split produces {split}, but the requested view type is {requested}. A view that reaches \
         less far than the buffer it indexes is an `unpad`, not a tile"
    )]
    UnexpectedView {
        /// The view produced by the split.
        split: Mapping,
        /// The requested tile view.
        requested: Mapping,
    },
}
