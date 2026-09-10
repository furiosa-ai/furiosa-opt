pub(super) fn sequential_i32(len: usize) -> Vec<i32> {
    (0..len).map(|value| value as i32).collect()
}

pub(super) fn reduce_scattered(placements: usize, shards: usize, rows: usize, columns: usize) -> Vec<i32> {
    (0..placements)
        .flat_map(|target| {
            (0..rows).flat_map(move |row| {
                (0..columns).map(move |column| {
                    (0..placements)
                        .map(|source| (((source * shards + target) * rows + row) * columns + column) as i32)
                        .sum()
                })
            })
        })
        .collect()
}

pub(super) fn all_reduced(placements: usize, rows: usize, columns: usize) -> Vec<i32> {
    (0..placements)
        .flat_map(|_| {
            (0..rows).flat_map(move |row| {
                (0..columns).map(move |column| {
                    (0..placements)
                        .map(|source| ((source * rows + row) * columns + column) as i32)
                        .sum()
                })
            })
        })
        .collect()
}

pub(super) fn gathered_shards(replicas: usize, shards: usize, rows: usize, columns: usize) -> Vec<i32> {
    (0..replicas)
        .flat_map(|_| {
            (0..shards).flat_map(move |shard| {
                (0..rows).flat_map(move |row| {
                    (0..columns).map(move |column| ((shard * rows + row) * columns + column) as i32)
                })
            })
        })
        .collect()
}

pub(super) fn all_reduced_shards(
    replicas: usize,
    reductions: usize,
    shards: usize,
    rows: usize,
    columns: usize,
) -> Vec<i32> {
    (0..replicas)
        .flat_map(|_| {
            (0..shards).flat_map(move |shard| {
                (0..rows).flat_map(move |row| {
                    (0..columns).map(move |column| {
                        (0..reductions)
                            .map(|source| (((source * shards + shard) * rows + row) * columns + column) as i32)
                            .sum()
                    })
                })
            })
        })
        .collect()
}
