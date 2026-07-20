//! Mapping-walk helpers shared by the storages.

use furiosa_mapping::{Index, Mapping, MappingExt, MappingIter};

/// `m.iter_positions()`: the natural self-walk — a mapping's own positions from the origin, padding included.
/// Collapses the recurring `m.iter(&m.axes(), &Index::new(), true)` idiom to one call. Yields one
/// `Option<usize>` per position, in position order (`None` for a padding position).
pub(crate) trait MappingPositions {
    fn iter_positions(&self) -> MappingIter;
}

impl MappingPositions for Mapping {
    fn iter_positions(&self) -> MappingIter {
        self.iter(&self.axes(), &Index::new(), true)
    }
}
