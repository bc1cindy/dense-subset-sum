//! Per-sub-tx W(E) estimates from CJA mapping enumeration.

use std::collections::HashMap;

use super::exclude_values;
use crate::count::sparse_conv::Goldilocks;
use crate::count::sumset::{Count, GradedSumset};
use crate::harness::vs_cja::mappings;
use crate::{Transaction, log_w_for_e_sat};

/// Two-sided ambiguity: `log_w_combined = log_w_lookup_inputs + log_w_lookup_outputs`
/// tracks `ln(count)` for balanced Maurer 2-partition mappings.
#[derive(Debug, Clone)]
pub struct SubTxEstimate {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
    pub balance: u64,
    pub log_sasamoto_approx: Option<f64>,
    /// Input-side lookup count.
    pub log_w_lookup: Option<f64>,
    pub log_w_lookup_outputs: Option<f64>,
    pub log_w_combined: Option<f64>,
    /// How many non-derived mappings contain this exact sub-tx.
    pub count: usize,
}

/// Estimate W for each UNIQUE sub-tx across non-derived mappings. Dedup key is
/// (sorted inputs, sorted outputs); `count` = mappings containing it.
/// Uses canonical Maurer framing: A = other inputs, E = Σ inputs of this sub-tx.
pub fn estimate_sub_txs(tx: &Transaction, knee: usize) -> Vec<SubTxEstimate> {
    let all_mappings = mappings::enumerate_mappings(tx);
    let non_derived = mappings::non_derived_mappings(&all_mappings);

    let mut groups: HashMap<(Vec<u64>, Vec<u64>), usize> = HashMap::new();
    for mapping in &non_derived {
        for (in_set, out_set) in mapping.input_sets.iter().zip(mapping.output_sets.iter()) {
            let mut ins = in_set.clone();
            ins.sort();
            let mut outs = out_set.clone();
            outs.sort();
            *groups.entry((ins, outs)).or_insert(0) += 1;
        }
    }

    let mut estimates = Vec::with_capacity(groups.len());
    for ((ins, outs), count) in groups {
        let balance: u64 = ins.iter().sum();
        let other_coins = exclude_values(&tx.inputs, &ins);
        let target = balance;

        let log_sasamoto_approx = sasamoto_finite(&other_coins, target);

        let log_w_lookup = log_lookup_w(&other_coins, target, knee);
        let other_outputs = exclude_values(&tx.outputs, &outs);
        let log_w_lookup_outputs = log_lookup_w(&other_outputs, balance, knee);

        let log_w_combined = match (log_w_lookup, log_w_lookup_outputs) {
            (Some(li), Some(lo)) if li.is_finite() && lo.is_finite() => Some(li + lo),
            _ => None,
        };

        estimates.push(SubTxEstimate {
            inputs: ins,
            outputs: outs,
            balance,
            log_sasamoto_approx,
            log_w_lookup,
            log_w_lookup_outputs,
            log_w_combined,
            count,
        });
    }
    estimates
}

fn log_lookup_w(set: &[u64], target: u64, knee: usize) -> Option<f64> {
    if set.is_empty() || target == 0 {
        return None;
    }
    let s: GradedSumset<Goldilocks> =
        GradedSumset::bounded(set, &[target], knee.min(set.len()).max(0));
    let visible = match s.count_total(target) {
        Count::Confirmed(n) | Count::Truncated(n) => n,
        Count::Absent | Count::Unknown => return Some(f64::NEG_INFINITY),
    };
    if visible == 0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some(f64::from(visible).ln())
    }
}

fn sasamoto_finite(set: &[u64], target: u64) -> Option<f64> {
    if set.is_empty() || target == 0 {
        return None;
    }
    let sum: u64 = set.iter().sum();
    if target >= sum {
        return None;
    }
    let lw = log_w_for_e_sat(set, target);
    if lw.is_finite() { Some(lw) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    #[test]
    fn test_sub_tx_estimates_maurer() {
        let tx = fixtures::maurer_fig2();
        let estimates = estimate_sub_txs(&tx, 4);

        assert!(
            !estimates.is_empty(),
            "should produce sub-tx estimates for Maurer Fig. 2"
        );

        for est in &estimates {
            assert_eq!(
                est.inputs.iter().sum::<u64>(),
                est.outputs.iter().sum::<u64>(),
                "sub-tx should be balanced"
            );
        }
    }

    #[test]
    fn test_sub_tx_estimate_combined_is_sum_of_sides() {
        let tx = fixtures::equal_denominations();
        let estimates = estimate_sub_txs(&tx, 4);

        let mut saw_both = false;
        for e in &estimates {
            if let (Some(li), Some(lo), Some(c)) =
                (e.log_w_lookup, e.log_w_lookup_outputs, e.log_w_combined)
                && li.is_finite()
                && lo.is_finite()
            {
                assert!(
                    (c - (li + lo)).abs() < 1e-9,
                    "combined must equal sum of input- and output-side logs: {} vs {}+{}",
                    c,
                    li,
                    lo
                );
                saw_both = true;
            }
        }
        assert!(
            saw_both,
            "expected at least one sub-tx with finite bilateral log_w in equal_denominations"
        );
    }

    #[test]
    fn test_estimate_sub_txs_dedups_identical_sub_txs() {
        let tx = fixtures::maurer_fig2();
        let estimates = estimate_sub_txs(&tx, 4);

        let mut seen = std::collections::HashSet::new();
        for e in &estimates {
            let key = (e.inputs.clone(), e.outputs.clone());
            assert!(
                seen.insert(key),
                "duplicate sub-tx returned: {:?} -> {:?}",
                e.inputs,
                e.outputs
            );
            assert!(e.count >= 1);
        }
    }
}
