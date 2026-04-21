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

#[derive(Debug, Clone)]
pub struct MappingMetrics {
    pub total_mappings: usize,
    pub non_derived_count: usize,
    /// log₂(non_derived_count), or 0 if count ≤ 1.
    pub entropy: f64,
    /// (input_idx, output_idx) pairs that share a sub-tx in every non-derived mapping.
    pub deterministic_links: Vec<(usize, usize)>,
}

/// Returns `log₂(|non-derived mappings|)`, or `None` when the tx exceeds `max_coins`
/// (enumeration is exponential — impractical past ~26 coins).
pub fn boltzmann_entropy(tx: &Transaction, max_coins: usize) -> Option<f64> {
    if tx.inputs.len() + tx.outputs.len() > max_coins {
        return None;
    }
    Some(analyze(tx).entropy)
}

pub fn analyze(tx: &Transaction) -> MappingMetrics {
    let all = enumerate_mappings(tx);
    let non_derived = non_derived_mappings(&all);
    let non_derived_count = non_derived.len();

    let entropy = if non_derived_count <= 1 {
        0.0
    } else {
        (non_derived_count as f64).log2()
    };

    // CJA partitions carry values not indices, so match by (value, nth occurrence).
    let mut deterministic_links = Vec::new();
    if !non_derived.is_empty() {
        for (i_idx, &i_val) in tx.inputs.iter().enumerate() {
            let i_occurrence = tx.inputs[..i_idx].iter().filter(|&&v| v == i_val).count();

            for (o_idx, &o_val) in tx.outputs.iter().enumerate() {
                let o_occurrence = tx.outputs[..o_idx].iter().filter(|&&v| v == o_val).count();

                let always_together = non_derived.iter().all(|m| {
                    let i_sub = find_occurrence_in_partition(&m.input_sets, i_val, i_occurrence);
                    let o_sub = find_occurrence_in_partition(&m.output_sets, o_val, o_occurrence);
                    i_sub.is_some() && i_sub == o_sub
                });
                if always_together {
                    deterministic_links.push((i_idx, o_idx));
                }
            }
        }
    }

    MappingMetrics {
        total_mappings: all.len(),
        non_derived_count,
        entropy,
        deterministic_links,
    }
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
    if a.num_sub_txs() != b.num_sub_txs() {
        return false;
    }

    let mut a_parts: Vec<(Vec<u64>, Vec<u64>)> = a
        .input_sets
        .iter()
        .zip(a.output_sets.iter())
        .map(|(i, o)| {
            let mut i = i.clone();
            let mut o = o.clone();
            i.sort();
            o.sort();
            (i, o)
        })
        .collect();
    a_parts.sort();

    let mut b_parts: Vec<(Vec<u64>, Vec<u64>)> = b
        .input_sets
        .iter()
        .zip(b.output_sets.iter())
        .map(|(i, o)| {
            let mut i = i.clone();
            let mut o = o.clone();
            i.sort();
            o.sort();
            (i, o)
        })
        .collect();
    b_parts.sort();

    a_parts == b_parts
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
    fn test_maurer_fig2_metrics() {
        let tx = fixtures::maurer_fig2();
        let metrics = analyze(&tx);
        assert!(metrics.total_mappings >= 2);
        assert!(metrics.non_derived_count >= 1);
        assert!(metrics.entropy >= 0.0);
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
    fn test_boltzmann_entropy_cap() {
        let tx = Transaction::new(vec![1; 40], vec![1; 40]);
        assert!(boltzmann_entropy(&tx, 26).is_none());
    }

    #[test]
    fn test_boltzmann_entropy_trivial_is_zero() {
        let tx = Transaction::new(vec![100], vec![100]);
        assert_eq!(boltzmann_entropy(&tx, 26), Some(0.0));
    }

    #[test]
    fn test_boltzmann_entropy_equal_denoms_positive() {
        let tx = fixtures::equal_denominations();
        let h = boltzmann_entropy(&tx, 26).expect("fits under cap");
        assert!(h > 0.0, "equal denominations should have H > 0, got {}", h);
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
    fn test_deterministic_links_with_duplicates() {
        let tx = fixtures::equal_denominations();
        let metrics = analyze(&tx);
        for &(i, o) in &metrics.deterministic_links {
            assert!(i < tx.inputs.len());
            assert!(o < tx.outputs.len());
        }
    }

    #[test]
    fn test_find_occurrence_in_partition() {
        let sets = vec![vec![100, 200], vec![100]];
        assert_eq!(super::find_occurrence_in_partition(&sets, 100, 0), Some(0));
        assert_eq!(super::find_occurrence_in_partition(&sets, 100, 1), Some(1));
        assert_eq!(super::find_occurrence_in_partition(&sets, 200, 0), Some(0));
        assert_eq!(super::find_occurrence_in_partition(&sets, 100, 2), None);
    }
}
