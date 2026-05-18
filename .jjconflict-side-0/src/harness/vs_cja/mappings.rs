//! Ground-truth mapping enumeration via CJA (Maurer et al.).

use coinjoin_analyzer::{
    Partition, PartitionsSubsetSumsFilter, SubsetSumsFilter, SumFilteredPartitionIterator,
};

use crate::Transaction;

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
    if tx.inputs.is_empty() || tx.outputs.is_empty() {
        return vec![];
    }

    // CJA's BloomFilter panics on sets with < 2 elements.
    if tx.inputs.len() == 1 || tx.outputs.len() == 1 {
        if tx.input_sum() == tx.output_sum() {
            return vec![Mapping {
                input_sets: vec![tx.inputs.clone()],
                output_sets: vec![tx.outputs.clone()],
            }];
        } else {
            return vec![];
        }
    }

    let out_filter = SubsetSumsFilter::new(&tx.outputs);
    let in_partitions: Vec<Partition> =
        SumFilteredPartitionIterator::new(tx.inputs.clone(), &out_filter).collect();

    if in_partitions.is_empty() {
        return vec![];
    }

    let in_parts_filter = PartitionsSubsetSumsFilter::new(&in_partitions);
    let out_partitions: Vec<Partition> =
        SumFilteredPartitionIterator::new(tx.outputs.clone(), &in_parts_filter).collect();

    let mut mappings = Vec::new();
    for in_partition in &in_partitions {
        for out_partition in &out_partitions {
            if partitions_match(in_partition, out_partition) {
                let (aligned_in, aligned_out) = align_partitions(in_partition, out_partition);
                mappings.push(Mapping {
                    input_sets: aligned_in,
                    output_sets: aligned_out,
                });
            }
        }
    }

    mappings
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
    mappings
        .iter()
        .filter(|m| !is_derived(m, mappings))
        .cloned()
        .collect()
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

fn align_partitions(inputs: &Partition, outputs: &Partition) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    let mut out_remaining: Vec<(u64, Vec<u64>)> = outputs
        .iter()
        .map(|s| (s.iter().sum(), s.clone()))
        .collect();

    let mut aligned_in = Vec::new();
    let mut aligned_out = Vec::new();

    for in_set in inputs {
        let in_sum: u64 = in_set.iter().sum();
        if let Some(pos) = out_remaining.iter().position(|(s, _)| *s == in_sum) {
            aligned_in.push(in_set.clone());
            aligned_out.push(out_remaining.swap_remove(pos).1);
        }
    }

    (aligned_in, aligned_out)
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
}
