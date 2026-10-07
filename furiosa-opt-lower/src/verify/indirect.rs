//! Where an indirect operation's tensors sit, and what its index may be.

use furiosa_mapping::{Mapping, MappingExt, PaddingKind};

/// Cluster, PE, and element placement of an SPM tensor, outermost first.
#[derive(Debug, Clone, Copy)]
pub struct SpmPlacement<'a> {
    pub cluster: &'a Mapping,
    pub pe: &'a Mapping,
    pub element: &'a Mapping,
}

/// Where a data-memory tensor's positions sit, outermost first.
#[derive(Debug, Clone, Copy)]
pub struct DmPlacement<'a> {
    pub cluster: &'a Mapping,
    pub slice: &'a Mapping,
    pub element: &'a Mapping,
}

impl DmPlacement<'_> {
    /// The single mapping the three axes form together.
    pub(super) fn to_mapping(self) -> Mapping {
        Mapping::pairs([self.cluster.clone(), self.slice.clone(), self.element.clone()])
    }
}

/// A mapping with padding cannot represent a dense index list.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("index list {list} must have one entry per position it spans, with no padding")]
pub struct IndexPaddingError {
    pub list: Mapping,
}

/// Where an indirect index lives, which is also the domain its entries key.
#[derive(Debug, Clone, Copy)]
pub enum IndexMapping<'a> {
    /// Byte offsets held in an HBM tensor, whose element mapping is the index domain.
    ByteOffsets(&'a Mapping),
    /// Unscaled positions held in an SPM tensor, whose live axes are the index domain.
    SpmPositions(SpmPlacement<'a>),
}

impl IndexMapping<'_> {
    /// The domain the index carries: the list, with the cluster in front where each cluster holds
    /// its own.
    ///
    /// A cluster that holds one copy names nothing, the way the PE placement never does. Every
    /// other cluster gives each of its positions a list, and joins the domain as it is written.
    pub fn domain(self) -> Result<Mapping, IndexPaddingError> {
        match self {
            Self::ByteOffsets(offsets) => {
                ensure_unpadded(offsets)?;
                Ok(offsets.clone().normalize())
            }
            Self::SpmPositions(index) => {
                ensure_unpadded(index.element)?;
                let list = index.element.clone().normalize();
                if holds_one_copy(index.cluster) {
                    return Ok(list);
                }
                Ok(index.cluster.clone().pair(list).normalize())
            }
        }
    }
}

/// Checks that an index list has a value at every position.
fn ensure_unpadded(list: &Mapping) -> Result<(), IndexPaddingError> {
    if list.has_no_padding() {
        return Ok(());
    }
    Err(IndexPaddingError { list: list.clone() })
}

/// Why an SPM-resident indirect index cannot be consumed by indirect DMA.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpmIndirectIndexError {
    #[error(
        "SPM indirect index Pe mapping {pe} must keep the entire list on PE 0 (`m![1 # {pe_count}]`)",
        pe_count = .pe.size()
    )]
    PePlacement { pe: Mapping },
}

/// Why an index cannot be read from the chips the data sits on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "index chip mapping {index} must either be broadcast or match the data's {data}; \
     there is no inter-chip indirect DMA, so a chip reads only the list its own chip holds"
)]
pub struct IndexChipError {
    pub index: Mapping,
    pub data: Mapping,
}

/// Checks that the index chip axis matches the data or broadcasts across it.
pub fn validate_index_chip(index: &Mapping, data: &Mapping) -> Result<(), IndexChipError> {
    if index.size() == data.size() && (index.is_broadcast() || index.normalize() == data.normalize()) {
        return Ok(());
    }
    Err(IndexChipError {
        index: index.clone(),
        data: data.clone(),
    })
}

/// Whether a placement holds one copy of what it carries rather than a share per position.
///
/// One live position padded across the hardware's, or the same cell broadcast to all of them.
/// Either way the placement names nothing of its own, so an index it holds is one list.
pub fn holds_one_copy(placement: &Mapping) -> bool {
    // `Top` leaves the other positions untouched; `Bottom` would mark them unreadable.
    let one = Mapping::identity()
        .padding(placement.size(), PaddingKind::Top)
        .normalize();
    placement.is_broadcast() || placement.clone().normalize() == one
}

/// Checks that an SPM indirect index resides entirely on PE 0.
pub fn validate_spm_indirect_index(pe: &Mapping) -> Result<(), SpmIndirectIndexError> {
    // `Top` leaves other PEs untouched; `Bottom` would mark them unreadable.
    let pe_zero = Mapping::identity().padding(pe.size(), PaddingKind::Top).normalize();
    if pe.clone().normalize() == pe_zero {
        return Ok(());
    }
    Err(SpmIndirectIndexError::PePlacement { pe: pe.clone() })
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![K = 4, CL = 2, V0 = 2, V1 = 4];

    fn spm_domain(cluster: &Mapping, pe: &Mapping, element: &Mapping) -> Mapping {
        IndexMapping::SpmPositions(SpmPlacement { cluster, pe, element })
            .domain()
            .unwrap()
    }

    #[test]
    fn the_domain_is_the_axes_the_index_names() {
        let pe = <m![1 # 4]>::to_value();
        assert_eq!(
            spm_domain(&<m![2]>::to_value(), &pe, &<m![V0]>::to_value()),
            <m![V0]>::to_value()
        );
        assert_eq!(
            spm_domain(&<m![CL]>::to_value(), &pe, &<m![V0]>::to_value()),
            <m![CL, V0]>::to_value()
        );
        // The list keeps its own order; rebuilding it from `axes()` would not.
        let list = <m![V0, V1]>::to_value();
        assert_eq!(IndexMapping::ByteOffsets(&list).domain().unwrap(), list.normalize());
        // Padding leaves cells no entry lives in, so it is refused rather than dropped.
        let padded = <m![1 # 64, V0, V1]>::to_value();
        assert!(IndexMapping::ByteOffsets(&padded).domain().is_err());
    }

    /// A replicated list is one list, so a broadcast cluster contributes no domain axis, while a
    /// placed one gives each cluster its own positions.
    #[test]
    fn a_broadcast_cluster_adds_no_domain_axis() {
        let (pe, element) = (<m![1 # 4]>::to_value(), <m![V0]>::to_value());
        let domain_of = |cluster: &Mapping| spm_domain(cluster, &pe, &element);

        assert_eq!(domain_of(&<m![2]>::to_value()), <m![V0]>::to_value().normalize());
        assert_eq!(domain_of(&<m![1 # 2]>::to_value()), <m![V0]>::to_value().normalize());
        assert_eq!(domain_of(&<m![CL]>::to_value()), <m![CL, V0]>::to_value().normalize());
    }

    #[test]
    fn accepts_an_index_kept_on_pe_zero() {
        validate_spm_indirect_index(&<m![1 # 4]>::to_value()).unwrap();
        validate_spm_indirect_index(&<m![1]>::to_value()).unwrap();
    }

    #[test]
    fn rejects_noncanonical_pe_placements() {
        let placements = [
            <m![4]>::to_value(),
            <m![K % 4]>::to_value(),
            <m![1 #{!} 4]>::to_value(),
            Mapping::Broadcast { size: 4 },
        ];
        for pe in placements {
            assert_eq!(
                validate_spm_indirect_index(&pe),
                Err(SpmIndirectIndexError::PePlacement { pe: pe.clone() }),
            );
        }
    }
}
