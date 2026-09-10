use furiosa_mapping::{Cell, Mapping, MappingExt};

/// Source placement and slice indices assigned to one SRAM destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SramRedistributeEntry {
    pub source_chip: usize,
    pub source_cluster: usize,
    pub slice_indices: Vec<usize>,
}

/// Why a slice or shuffle pattern cannot name the requested placements.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SlicePatternError {
    #[error("slice: expected one index per target ({num_targets}), got {}: {indices:?}", indices.len())]
    WrongIndexCount { indices: Vec<usize>, num_targets: usize },
    #[error("slice: index {index} for target {target} is out of bounds for axis {axis}")]
    InvalidIndex { target: usize, index: usize, axis: Mapping },
    #[error(
        "chip shuffle: expected one source per chip ({num_chips}), got {}: {source_chips:?}",
        source_chips.len()
    )]
    WrongSourceCount { source_chips: Vec<usize>, num_chips: usize },
    #[error("chip shuffle: source {source_chip} for target {target} is outside 0..{num_chips}")]
    InvalidSource {
        target: usize,
        source_chip: usize,
        num_chips: usize,
    },
    #[error("chip shuffle: source chip {source_chip} occurs more than once in {source_chips:?}")]
    DuplicateSource {
        source_chip: usize,
        source_chips: Vec<usize>,
    },
    #[error(
        "SRAM redistribution: target {target} needs {} slice indices, got {}",
        slice_axes.len(),
        slice_indices.len()
    )]
    WrongCoordinateCount {
        target: usize,
        slice_indices: Vec<usize>,
        slice_axes: Vec<Mapping>,
    },
    #[error(
        "SRAM redistribution: expected one source per destination cluster ({num_targets}), got {}",
        entries.len()
    )]
    WrongPlacementCount {
        entries: Vec<SramRedistributeEntry>,
        num_targets: usize,
    },
    #[error(
        "SRAM redistribution: source ({source_chip}, {source_cluster}) for target {target} is outside {num_chips} chips x {num_clusters} clusters"
    )]
    InvalidPlacement {
        target: usize,
        source_chip: usize,
        source_cluster: usize,
        num_chips: usize,
        num_clusters: usize,
    },
    #[error(
        "SRAM redistribution: live target ({target_chip}, {target_cluster}) cannot read a padded source placement ({source_chip}, {source_cluster}) in Chip {chip_mapping} and Cluster {cluster_mapping}"
    )]
    PaddedSourcePlacement {
        target_chip: usize,
        target_cluster: usize,
        source_chip: usize,
        source_cluster: usize,
        chip_mapping: Mapping,
        cluster_mapping: Mapping,
    },
    #[error("SRAM redistribution: source ({source_chip}, {source_cluster}) occurs more than once")]
    DuplicatePlacement { source_chip: usize, source_cluster: usize },
    #[error("SRAM redistribution: destination chip {target_chip} must read one source chip, got {source_chips:?}")]
    NonUniformSourceChip {
        target_chip: usize,
        source_chips: Vec<usize>,
    },
    #[error(
        "SRAM redistribution: destination cluster {target_cluster} must read one source cluster, got {source_clusters:?}"
    )]
    NonUniformSourceCluster {
        target_cluster: usize,
        source_clusters: Vec<usize>,
    },
}

/// A chip/cluster placement in the chip-major redistribution table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterPlacement {
    pub chip: usize,
    pub cluster: usize,
}

impl ClusterPlacement {
    /// Decodes a chip-major redistribution table index into its chip and cluster placement.
    pub fn from_target_index(index: usize, num_clusters: usize) -> Self {
        debug_assert!(num_clusters > 0);
        Self {
            chip: index / num_clusters,
            cluster: index % num_clusters,
        }
    }

    /// Returns the chip-major table index for this placement.
    pub fn to_target_index(self, num_clusters: usize) -> usize {
        debug_assert!(num_clusters > 0);
        self.chip * num_clusters + self.cluster
    }
}

/// Validates one slice index per destination.
pub fn validate_slice_indices(
    indices_by_target: &[usize],
    num_targets: usize,
    axis: &Mapping,
) -> Result<(), SlicePatternError> {
    if indices_by_target.len() != num_targets {
        return Err(SlicePatternError::WrongIndexCount {
            indices: indices_by_target.to_vec(),
            num_targets,
        });
    }
    for (target, &index) in indices_by_target.iter().enumerate() {
        ensure_index_in_bounds(axis, target, index)?;
    }
    Ok(())
}

/// Validates the source-chip permutation for a chip shuffle.
pub fn validate_chip_shuffle(source_chips_by_target: &[usize], num_chips: usize) -> Result<(), SlicePatternError> {
    if source_chips_by_target.len() != num_chips {
        return Err(SlicePatternError::WrongSourceCount {
            source_chips: source_chips_by_target.to_vec(),
            num_chips,
        });
    }
    let mut seen = vec![false; num_chips];
    for (target, &source_chip) in source_chips_by_target.iter().enumerate() {
        if source_chip >= num_chips {
            return Err(SlicePatternError::InvalidSource {
                target,
                source_chip,
                num_chips,
            });
        }
        if seen[source_chip] {
            return Err(SlicePatternError::DuplicateSource {
                source_chip,
                source_chips: source_chips_by_target.to_vec(),
            });
        }
        seen[source_chip] = true;
    }
    Ok(())
}

/// Validates the source table for an SRAM redistribution.
pub fn validate_sram_redistribution(
    entries_by_target: &[SramRedistributeEntry],
    chip_mapping: &Mapping,
    cluster_mapping: &Mapping,
    slice_axes: &[Mapping],
) -> Result<(), SlicePatternError> {
    let num_chips = chip_mapping.size();
    let num_clusters = cluster_mapping.size();
    debug_assert!(num_clusters > 0);
    let num_targets = num_chips * num_clusters;
    if entries_by_target.len() != num_targets {
        return Err(SlicePatternError::WrongPlacementCount {
            entries: entries_by_target.to_vec(),
            num_targets,
        });
    }

    let mut seen = vec![false; num_targets];
    for (target, source) in entries_by_target.iter().enumerate() {
        if source.source_chip >= num_chips || source.source_cluster >= num_clusters {
            return Err(SlicePatternError::InvalidPlacement {
                target,
                source_chip: source.source_chip,
                source_cluster: source.source_cluster,
                num_chips,
                num_clusters,
            });
        }
        let source_index = ClusterPlacement {
            chip: source.source_chip,
            cluster: source.source_cluster,
        }
        .to_target_index(num_clusters);
        if seen[source_index] {
            return Err(SlicePatternError::DuplicatePlacement {
                source_chip: source.source_chip,
                source_cluster: source.source_cluster,
            });
        }
        seen[source_index] = true;

        let ClusterPlacement {
            chip: target_chip,
            cluster: target_cluster,
        } = ClusterPlacement::from_target_index(target, num_clusters);
        let target_is_live = matches!(chip_mapping.index(target_chip), Cell::Index(_))
            && matches!(cluster_mapping.index(target_cluster), Cell::Index(_));
        let source_is_live = matches!(chip_mapping.index(source.source_chip), Cell::Index(_))
            && matches!(cluster_mapping.index(source.source_cluster), Cell::Index(_));
        if target_is_live && !source_is_live {
            return Err(SlicePatternError::PaddedSourcePlacement {
                target_chip,
                target_cluster,
                source_chip: source.source_chip,
                source_cluster: source.source_cluster,
                chip_mapping: chip_mapping.clone(),
                cluster_mapping: cluster_mapping.clone(),
            });
        }

        if source.slice_indices.len() != slice_axes.len() {
            return Err(SlicePatternError::WrongCoordinateCount {
                target,
                slice_indices: source.slice_indices.clone(),
                slice_axes: slice_axes.to_vec(),
            });
        }
        for (axis, &index) in slice_axes.iter().zip(&source.slice_indices) {
            ensure_index_in_bounds(axis, target, index)?;
        }
    }

    for (target_chip, cluster_sources) in entries_by_target.chunks(num_clusters).enumerate() {
        let source_chips = cluster_sources
            .iter()
            .map(|source| source.source_chip)
            .collect::<Vec<_>>();
        if let Some((&first, rest)) = source_chips.split_first()
            && rest.iter().any(|source| *source != first)
        {
            return Err(SlicePatternError::NonUniformSourceChip {
                target_chip,
                source_chips,
            });
        }
    }
    for target_cluster in 0..num_clusters {
        let source_clusters = entries_by_target
            .iter()
            .skip(target_cluster)
            .step_by(num_clusters)
            .map(|source| source.source_cluster)
            .collect::<Vec<_>>();
        if let Some((&first, rest)) = source_clusters.split_first()
            && rest.iter().any(|source| *source != first)
        {
            return Err(SlicePatternError::NonUniformSourceCluster {
                target_cluster,
                source_clusters,
            });
        }
    }
    Ok(())
}

fn ensure_index_in_bounds(axis: &Mapping, target: usize, index: usize) -> Result<(), SlicePatternError> {
    if matches!(axis.index(index), Cell::Index(_) | Cell::Padding(_)) {
        Ok(())
    } else {
        Err(SlicePatternError::InvalidIndex {
            target,
            index,
            axis: axis.clone(),
        })
    }
}
