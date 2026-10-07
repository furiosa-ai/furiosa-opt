use furiosa_mapping::*;

fn canonical(cell: Cell) -> Result<Vec<(Ident, usize)>, PaddingKind> {
    cell.finalize(PaddingKind::Bottom)
        .into_result()
        .map(|index| index.into_iter().collect())
}

fn oracle_strides(table: &Mapping, axis: &Mapping) -> Vec<usize> {
    let axis_cells = (0..axis.size())
        .map(|position| canonical(axis.index(position)))
        .collect::<Vec<_>>();
    (1..=table.size() / axis.size())
        .filter(|stride| {
            axis_cells
                .iter()
                .enumerate()
                .all(|(position, cell)| cell == &canonical(table.index(position * stride)))
        })
        .collect()
}

fn next(state: &mut u64, upper: usize) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as usize) % upper
}

fn axis(symbol: Ident, size: usize, padding: usize, kind: PaddingKind) -> Mapping {
    Mapping::Symbol { symbol, size }.padding(padding, kind)
}

fn divisors(value: usize) -> Vec<usize> {
    (1..=value).filter(|divisor| value.is_multiple_of(*divisor)).collect()
}

fn assert_matches_oracle(case: usize, table: &Mapping, candidate: &Mapping) {
    let expected = oracle_strides(table, candidate);
    let actual = table.find_axis(candidate);
    let context = || {
        format!(
            "case {case}: candidate {} in {}, oracle={expected:?}",
            candidate.normalize(),
            table.normalize(),
        )
    };
    // Address enumeration checks acceptance; hand-written tests cover error variants.
    match expected.as_slice() {
        [stride] => assert_eq!(actual, Ok(*stride), "{}", context()),
        _ => assert!(actual.is_err(), "{}", context()),
    }
}

#[test]
fn fuzz_find_axis_against_enumerated_addresses() {
    let symbols = [Ident::A, Ident::B, Ident::C, Ident::D, Ident::E, Ident::F];
    let kinds = [PaddingKind::Top, PaddingKind::Bottom, PaddingKind::Zero];
    let mut state = 0x8db2_12f3_a93c_6d77;

    for case in 0..10_000 {
        let rank = 2 + next(&mut state, 3);
        let mut factors = Vec::with_capacity(rank);
        for &symbol in &symbols[..rank] {
            let size = 2 + next(&mut state, 3);
            let padding = size + next(&mut state, 2);
            factors.push(axis(symbol, size, padding, kinds[next(&mut state, kinds.len())]));
        }

        let table = Mapping::pairs(factors.iter().cloned());
        let begin = next(&mut state, rank);
        let end = begin + 1 + next(&mut state, rank - begin);
        let candidate = Mapping::pairs(factors[begin..end].iter().cloned());
        assert_matches_oracle(case, &table, &candidate);
        for axis in table.axes() {
            assert_matches_oracle(case, &table, &Mapping::from_terms([axis.to_term()]));
        }
    }
}

#[test]
fn fuzz_find_axis_in_factorized_and_mismatched_layouts() {
    let symbols = [Ident::A, Ident::B, Ident::C, Ident::D, Ident::E, Ident::F];
    let kinds = [PaddingKind::Top, PaddingKind::Bottom, PaddingKind::Zero];
    let mut state = 0xf527_6a21_05de_c8bb;

    for case in 0..10_000 {
        let rank = 2 + next(&mut state, 3);
        let mut factors = Vec::with_capacity(rank);
        for &symbol in &symbols[..rank] {
            let size = 2 + next(&mut state, 3);
            let padding = size + next(&mut state, 2);
            factors.push(axis(symbol, size, padding, kinds[next(&mut state, kinds.len())]));
        }

        let base = Mapping::pairs(factors.iter().cloned());
        let candidates = divisors(base.size());
        let divisor = candidates[next(&mut state, candidates.len())];
        let table = base.clone().stride(divisor).pair(base.clone().modulo(divisor));
        let begin = next(&mut state, rank);
        let mut picked = vec![factors[begin].clone()];
        match next(&mut state, 4) {
            0 => {
                let end = begin + 1 + next(&mut state, rank - begin);
                picked.extend(factors[begin + 1..end].iter().cloned());
            }
            1 if begin + 2 < rank => picked.push(factors[begin + 2].clone()),
            2 if begin + 1 < rank => picked.push(factors[begin + 1].clone()),
            _ => picked[0] = axis(Ident::Z, 2, 2, PaddingKind::Top),
        }
        if next(&mut state, 5) == 0 {
            picked.reverse();
        }
        let candidate = Mapping::pairs(picked);
        assert_matches_oracle(case, &table, &candidate);
    }
}

#[test]
#[ignore = "Locate does not compare padding kinds past the matched extent (backlog)"]
fn find_axis_rejects_a_padding_kind_mismatch_past_the_matched_extent() {
    // `C` is live 2 and padded to 4; asked for with the opposite padding kind, it reports stride 2.
    let table = Mapping::Symbol {
        symbol: Ident::E,
        size: 2,
    }
    .pair(axis(Ident::C, 2, 4, PaddingKind::Bottom))
    .pair(Mapping::Symbol {
        symbol: Ident::F,
        size: 2,
    });
    let candidate = axis(Ident::C, 2, 4, PaddingKind::Zero);

    // `Locate` truncates the segment before comparing pad kinds; exact sequencer matching is needed.
    assert!(table.find_axis(&candidate).is_err());
}
