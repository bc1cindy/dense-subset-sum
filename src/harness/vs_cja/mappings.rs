//! Exact mapping enumeration built on CJA partition iterators (Maurer et al.).

use coinjoin_analyzer::{
    Partition, PartitionsSubsetSumsFilter, SubsetSumsFilter, SumFilteredPartitionIterator,
};

use std::time::Instant;

use crate::Transaction;

type AlignedPartitions = (Partition, Partition);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapping {
    pub input_sets: Vec<Vec<u64>>,
    pub output_sets: Vec<Vec<u64>>,
}

impl Mapping {
    pub fn num_sub_txs(&self) -> usize {
        self.input_sets.len()
    }
}

/// Exponential in total coin count — practical up to ~25 coins.
pub fn enumerate_mappings(tx: &Transaction) -> Vec<Mapping> {
    enumerate_mappings_within(tx, None).expect("no deadline can not expire")
}

/// The same enumeration under a wall-clock deadline, `None` when it expires.
///
/// Coin count is a poor predictor of this cost: measured over one real provenance
/// walk, calls of the same total size ranged from 54ms to 69s, the slow one being
/// 12 inputs against 17 outputs of which nine shared a value. Repeated values
/// multiply the partitions, and neither the count, the density, nor the per-coin
/// weight separates the cheap case from the expensive one. A deadline bounds what
/// the caller actually cares about; a size limit only bounds a proxy for it.
///
/// Expiry is checked between partitions rather than inside the iterator, so the
/// bound is honoured to within one partition's work and never mid-structure.
pub fn enumerate_mappings_within(
    tx: &Transaction,
    deadline: Option<Instant>,
) -> Option<Vec<Mapping>> {
    let expired = || deadline.is_some_and(|d| Instant::now() >= d);
    if tx.inputs.is_empty() || tx.outputs.is_empty() {
        return Some(vec![]);
    }

    // CJA's BloomFilter panics on sets with < 2 elements.
    if tx.inputs.len() == 1 || tx.outputs.len() == 1 {
        if tx.input_sum() == tx.output_sum() {
            return Some(vec![Mapping {
                input_sets: vec![tx.inputs.clone()],
                output_sets: vec![tx.outputs.clone()],
            }]);
        } else {
            return Some(vec![]);
        }
    }

    let out_filter = SubsetSumsFilter::new(&tx.outputs);
    let mut in_partitions: Vec<Partition> = Vec::new();
    for partition in SumFilteredPartitionIterator::new(tx.inputs.clone(), &out_filter) {
        if expired() {
            return None;
        }
        in_partitions.push(partition);
    }

    if in_partitions.is_empty() {
        return Some(vec![]);
    }

    let in_parts_filter = PartitionsSubsetSumsFilter::new(&in_partitions);
    let mut out_partitions: Vec<Partition> = Vec::new();
    for partition in SumFilteredPartitionIterator::new(tx.outputs.clone(), &in_parts_filter) {
        if expired() {
            return None;
        }
        out_partitions.push(partition);
    }

    let mut mappings = Vec::new();
    for in_partition in &in_partitions {
        if expired() {
            return None;
        }
        for out_partition in &out_partitions {
            if partitions_match(in_partition, out_partition) {
                let alignments = align_partitions_within(in_partition, out_partition, deadline)?;
                for (input_sets, output_sets) in alignments {
                    mappings.push(Mapping {
                        input_sets,
                        output_sets,
                    });
                }
            }
        }
    }

    Some(mappings)
}

/// Derived = obtainable by merging two sub-txs of a mapping with one more sub-tx.
pub fn is_derived(mapping: &Mapping, all_mappings: &[Mapping]) -> bool {
    let k = mapping.num_sub_txs();
    for other in all_mappings {
        if other.num_sub_txs() != k + 1 {
            continue;
        }
        for i in 0..other.input_sets.len() {
            for j in (i + 1)..other.input_sets.len() {
                let merged = merge_sub_txs(other, i, j);
                if mapping_equivalent(mapping, &merged) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn non_derived_mappings(mappings: &[Mapping]) -> Vec<Mapping> {
    non_derived_mappings_within(mappings, None).expect("no deadline can not expire")
}

/// The same filter under a deadline, `None` when it expires.
///
/// This half is quadratic in the mapping count and can dominate the enumeration
/// that produced it, so a deadline covering only the enumeration would bound the
/// cheaper phase and leave the expensive one running.
pub fn non_derived_mappings_within(
    mappings: &[Mapping],
    deadline: Option<Instant>,
) -> Option<Vec<Mapping>> {
    let mut out = Vec::new();
    for mapping in mappings {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return None;
        }
        if !is_derived(mapping, mappings) {
            out.push(mapping.clone());
        }
    }
    Some(out)
}

/// Maurer/Boltzmann entropy in bits: `log₂(n_non_derived)`. Returns `0.0` when count ≤ 1
/// (single interpretation: no attacker uncertainty). Caller is responsible for capping
/// enumeration cost upstream.
#[must_use]
pub fn boltzmann_entropy(n_non_derived: usize) -> f64 {
    if n_non_derived <= 1 {
        0.0
    } else {
        (n_non_derived as f64).log2()
    }
}

/// `p_IO[i][o]` = fraction of `non_derived` mappings where the `i`-th input
/// (by position in `tx.inputs`, matching by `(value, n-th occurrence)`) and the
/// `o`-th output share a sub-transaction. Granularity per coin pair; complements
/// the global `n_non_derived` count.
///
/// Returns an `n_inputs × n_outputs` matrix of probabilities in `[0.0, 1.0]`.
/// All-zero rows/cols when `non_derived` is empty.
#[must_use]
pub fn pairwise_input_output_prob(non_derived: &[Mapping], tx: &Transaction) -> Vec<Vec<f64>> {
    let n_in = tx.inputs.len();
    let n_out = tx.outputs.len();
    let mut matrix = vec![vec![0.0f64; n_out]; n_in];
    if non_derived.is_empty() {
        return matrix;
    }

    let total = non_derived.len() as f64;
    for (i_idx, &i_val) in tx.inputs.iter().enumerate() {
        let i_occurrence = tx.inputs[..i_idx].iter().filter(|&&v| v == i_val).count();
        for (o_idx, &o_val) in tx.outputs.iter().enumerate() {
            let o_occurrence = tx.outputs[..o_idx].iter().filter(|&&v| v == o_val).count();
            let hits = non_derived
                .iter()
                .filter(|m| {
                    let i_sub = find_occurrence_in_partition(&m.input_sets, i_val, i_occurrence);
                    let o_sub = find_occurrence_in_partition(&m.output_sets, o_val, o_occurrence);
                    i_sub.is_some() && i_sub == o_sub
                })
                .count();
            matrix[i_idx][o_idx] = hits as f64 / total;
        }
    }
    matrix
}

/// Recognizes a large repeated-denomination transaction for which exact mapping enumeration is
/// deliberately skipped. This is only a tractability classification: it does not derive pairwise
/// probabilities or certify ambiguity.
#[must_use]
pub(crate) fn is_repeated_denomination_dense_case(
    inputs: &[u64],
    real_outputs: &[u64],
    min_coins: usize,
) -> bool {
    if inputs.len() + real_outputs.len() <= min_coins {
        return false;
    }
    use std::collections::HashMap;
    let mut mult: HashMap<u64, usize> = HashMap::new();
    for &v in real_outputs {
        *mult.entry(v).or_insert(0) += 1;
    }
    let repeated_denoms = mult.values().filter(|&&c| c >= 3).count();
    let covered: usize = mult.values().filter(|&&c| c >= 3).sum();
    // >=2 distinct denominations each repeated >=3x, covering >=half the outputs => coinjoin-dense.
    repeated_denoms >= 2 && covered * 2 >= real_outputs.len()
}

/// Pairs `(input_idx, output_idx)` linked in **every** non-derived mapping
/// (`p_IO == 1.0`). Each pair represents a coin with anonymity zero: the
/// attacker can pin it without ambiguity, even when `n_non_derived` is large.
#[must_use]
pub fn deterministic_links(non_derived: &[Mapping], tx: &Transaction) -> Vec<(usize, usize)> {
    if non_derived.is_empty() {
        return vec![];
    }
    let matrix = pairwise_input_output_prob(non_derived, tx);
    let mut out = Vec::new();
    for (i, row) in matrix.iter().enumerate() {
        for (o, &p) in row.iter().enumerate() {
            if p >= 1.0 {
                out.push((i, o));
            }
        }
    }
    out
}

fn find_occurrence_in_partition(sets: &[Vec<u64>], val: u64, occurrence: usize) -> Option<usize> {
    let mut seen = 0;
    for (s, set) in sets.iter().enumerate() {
        let count_in_set = set.iter().filter(|&&v| v == val).count();
        if seen + count_in_set > occurrence {
            return Some(s);
        }
        seen += count_in_set;
    }
    None
}

fn partitions_match(a: &Partition, b: &Partition) -> bool {
    let mut b_sums: Vec<u64> = b.iter().map(|s| s.iter().sum()).collect();
    for set_a in a {
        let sum_a: u64 = set_a.iter().sum();
        if let Some(pos) = b_sums.iter().position(|&s| s == sum_a) {
            b_sums.swap_remove(pos);
        } else {
            return false;
        }
    }
    true
}

fn align_partitions_within(
    inputs: &Partition,
    outputs: &Partition,
    deadline: Option<Instant>,
) -> Option<Vec<AlignedPartitions>> {
    fn visit(
        inputs: &Partition,
        outputs: &Partition,
        input_index: usize,
        used_outputs: &mut [bool],
        aligned_outputs: &mut Vec<Vec<u64>>,
        alignments: &mut Vec<AlignedPartitions>,
        deadline: Option<Instant>,
    ) -> Option<()> {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return None;
        }
        if input_index == inputs.len() {
            alignments.push((inputs.clone(), aligned_outputs.clone()));
            return Some(());
        }

        let input_sum: u64 = inputs[input_index].iter().sum();
        for (output_index, output_set) in outputs.iter().enumerate() {
            if !used_outputs[output_index] && output_set.iter().sum::<u64>() == input_sum {
                used_outputs[output_index] = true;
                aligned_outputs.push(output_set.clone());
                visit(
                    inputs,
                    outputs,
                    input_index + 1,
                    used_outputs,
                    aligned_outputs,
                    alignments,
                    deadline,
                )?;
                aligned_outputs.pop();
                used_outputs[output_index] = false;
            }
        }
        Some(())
    }

    let mut alignments = Vec::new();
    visit(
        inputs,
        outputs,
        0,
        &mut vec![false; outputs.len()],
        &mut Vec::with_capacity(inputs.len()),
        &mut alignments,
        deadline,
    )?;
    Some(alignments)
}

fn merge_sub_txs(m: &Mapping, i: usize, j: usize) -> Mapping {
    let mut input_sets = Vec::new();
    let mut output_sets = Vec::new();

    let mut merged_in = Vec::new();
    let mut merged_out = Vec::new();

    for (idx, (ins, outs)) in m.input_sets.iter().zip(m.output_sets.iter()).enumerate() {
        if idx == i || idx == j {
            merged_in.extend(ins.iter().copied());
            merged_out.extend(outs.iter().copied());
        } else {
            input_sets.push(ins.clone());
            output_sets.push(outs.clone());
        }
    }

    merged_in.sort();
    merged_out.sort();
    input_sets.push(merged_in);
    output_sets.push(merged_out);

    Mapping {
        input_sets,
        output_sets,
    }
}

fn mapping_equivalent(a: &Mapping, b: &Mapping) -> bool {
    a.num_sub_txs() == b.num_sub_txs() && canonical_parts(a) == canonical_parts(b)
}

fn canonical_parts(m: &Mapping) -> Vec<(Vec<u64>, Vec<u64>)> {
    let mut parts: Vec<(Vec<u64>, Vec<u64>)> = m
        .input_sets
        .iter()
        .zip(m.output_sets.iter())
        .map(|(i, o)| {
            let mut i = i.clone();
            let mut o = o.clone();
            i.sort();
            o.sort();
            (i, o)
        })
        .collect();
    parts.sort();
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    #[test]
    fn repeated_denomination_dense_case_is_only_a_classifier() {
        use super::is_repeated_denomination_dense_case;
        let inputs: Vec<u64> = (0..12).map(|k| 1_000_000 + k).collect();
        let mut outputs = vec![20_000; 8];
        outputs.extend(std::iter::repeat_n(2_097_152, 8));
        outputs.extend(std::iter::repeat_n(5_000_000, 4));
        // 20 outputs, 3 repeated denoms cover all
        assert!(is_repeated_denomination_dense_case(&inputs, &outputs, 15));
        assert!(
            !is_repeated_denomination_dense_case(&[500_000, 500_000], &[900_000, 90_000], 15),
            "payment must not fire"
        );
        let distinct: Vec<u64> = (1..=40).map(|k| k * 111_113).collect();
        assert!(
            !is_repeated_denomination_dense_case(&[1_000_000u64; 5], &distinct, 15),
            "distinct-large must not fire"
        );
        assert!(
            !is_repeated_denomination_dense_case(&[3, 5], &[8], 15),
            "small must not fire"
        );
    }

    #[test]
    fn test_maurer_fig2_mappings() {
        // Maurer Fig. 2: 2 mappings (1 non-derived + 1 derived).
        let tx = fixtures::maurer_fig2();
        let all = enumerate_mappings(&tx);

        for (i, m) in all.iter().enumerate() {
            eprintln!("Mapping {}: {} sub-txs", i, m.num_sub_txs());
            for (ins, outs) in m.input_sets.iter().zip(m.output_sets.iter()) {
                eprintln!(
                    "  {:?} -> {:?} (sum={})",
                    ins,
                    outs,
                    ins.iter().sum::<u64>()
                );
            }
        }

        assert_eq!(all.len(), 2);
        assert_eq!(non_derived_mappings(&all).len(), 1);

        for m in &all {
            for (ins, outs) in m.input_sets.iter().zip(m.output_sets.iter()) {
                assert_eq!(
                    ins.iter().sum::<u64>(),
                    outs.iter().sum::<u64>(),
                    "unbalanced sub-tx: {:?} vs {:?}",
                    ins,
                    outs
                );
            }
        }
    }

    #[test]
    fn test_equal_denominations_mappings() {
        let tx = fixtures::equal_denominations();
        assert!(enumerate_mappings(&tx).len() > 1);
    }

    #[test]
    fn test_trivial_single_participant() {
        let tx = Transaction::new(vec![100], vec![100]);
        let all = enumerate_mappings(&tx);
        assert_eq!(all.len(), 1);
        assert_eq!(non_derived_mappings(&all).len(), 1);
    }

    #[test]
    fn test_boltzmann_entropy_zero_for_unique_interpretation() {
        assert_eq!(boltzmann_entropy(0), 0.0);
        assert_eq!(boltzmann_entropy(1), 0.0);
    }

    #[test]
    fn test_boltzmann_entropy_is_log2() {
        assert_eq!(boltzmann_entropy(2), 1.0);
        assert_eq!(boltzmann_entropy(4), 2.0);
        assert_eq!(boltzmann_entropy(1024), 10.0);
    }

    #[test]
    fn test_partitions_match_basic() {
        let a: Partition = vec![vec![10, 20], vec![30]];
        let b: Partition = vec![vec![30], vec![15, 15]];
        assert!(partitions_match(&a, &b));

        let c: Partition = vec![vec![10], vec![20, 30]];
        assert!(!partitions_match(&a, &c));
    }

    #[test]
    fn test_pairwise_prob_maurer_fig2() {
        // 1 non-derived mapping: {(21,12)→(25,8), (36,28)→(50,14)}.
        // i₀=21,i₁=12 share sub-tx 0 with o₀=25,o₁=8.
        // i₂=36,i₃=28 share sub-tx 1 with o₂=50,o₃=14.
        let tx = fixtures::maurer_fig2();
        let all = enumerate_mappings(&tx);
        let nd = non_derived_mappings(&all);
        let p = pairwise_input_output_prob(&nd, &tx);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].len(), 4);
        assert_eq!(p[0][0], 1.0, "i0=21 ↔ o0=25 forced");
        assert_eq!(p[0][2], 0.0, "i0 never shares sub-tx with o2");
        assert_eq!(p[2][2], 1.0, "i2=36 ↔ o2=50 forced");
        assert_eq!(p[2][0], 0.0, "i2 never with o0");
    }

    #[test]
    fn test_deterministic_links_maurer_fig2() {
        let tx = fixtures::maurer_fig2();
        let nd = non_derived_mappings(&enumerate_mappings(&tx));
        let links = deterministic_links(&nd, &tx);
        // 4 pairs forced: (0,0), (0,1), (1,0), (1,1) on Alice side + (2,2),(2,3),(3,2),(3,3) on Bob side.
        assert_eq!(links.len(), 8);
    }

    #[test]
    fn test_pairwise_prob_empty_when_no_mappings() {
        let tx = Transaction::new(vec![1], vec![2]);
        let p = pairwise_input_output_prob(&[], &tx);
        assert_eq!(p, vec![vec![0.0]]);
        let links = deterministic_links(&[], &tx);
        assert!(links.is_empty());
    }

    #[test]
    fn test_deadline_expired_returns_none() {
        let tx = fixtures::equal_denominations();
        let past = Instant::now() - std::time::Duration::from_secs(1);
        assert!(enumerate_mappings_within(&tx, Some(past)).is_none());
        let all = enumerate_mappings(&tx);
        assert!(non_derived_mappings_within(&all, Some(past)).is_none());
    }

    #[test]
    fn test_generous_deadline_matches_unbounded() {
        let tx = fixtures::maurer_fig2();
        let far = Instant::now() + std::time::Duration::from_secs(60);
        let bounded = enumerate_mappings_within(&tx, Some(far)).unwrap();
        assert_eq!(bounded, enumerate_mappings(&tx));
        let nd = non_derived_mappings_within(&bounded, Some(far)).unwrap();
        assert_eq!(nd, non_derived_mappings(&bounded));
    }
}
