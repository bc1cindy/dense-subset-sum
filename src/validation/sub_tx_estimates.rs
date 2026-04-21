//! Per-sub-tx W(E) estimates from CJA mapping enumeration.

use std::collections::HashMap;

use super::exclude_values;
use crate::estimator::saddle_reliable;
use crate::mappings;
use crate::{Transaction, log_lookup_w, log_w_for_e_sat};

const DEFAULT_SADDLE_TAU: f64 = 0.5;

/// Two-sided ambiguity: `log_w_combined = log_w_lookup_inputs + log_w_lookup_outputs`
/// tracks `ln(count)` for balanced Maurer 2-partition mappings.
#[derive(Debug, Clone)]
pub struct SubTxEstimate {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
    pub balance: u64,
    pub log_w_sasamoto: Option<f64>,
    /// Input-side lookup count. Named `log_w_lookup` (not `_inputs`) for back-compat.
    pub log_w_lookup: Option<f64>,
    pub log_w_lookup_outputs: Option<f64>,
    pub log_w_combined: Option<f64>,
    /// How many non-derived mappings contain this exact sub-tx.
    pub count: usize,
    /// False when N_other < 20 or radix-like (asymptotic regime not reached).
    pub sasamoto_reliable: bool,
}

/// Estimate W for each UNIQUE sub-tx across non-derived mappings. Dedup key is
/// (sorted inputs, sorted outputs); `count` = mappings containing it.
/// Uses canonical Maurer framing: A = other inputs, E = Σ inputs of this sub-tx.
pub fn estimate_sub_txs(tx: &Transaction, lookup_k: usize) -> Vec<SubTxEstimate> {
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

        let log_w_sasamoto = if !other_coins.is_empty() && target > 0 {
            log_w_for_e_sat(&other_coins, target)
        } else {
            None
        };

        let log_w_lookup = if !other_coins.is_empty() && target > 0 {
            log_lookup_w(&other_coins, target, lookup_k)
        } else {
            None
        };

        let other_outputs = exclude_values(&tx.outputs, &outs);
        let log_w_lookup_outputs = if !other_outputs.is_empty() && balance > 0 {
            log_lookup_w(&other_outputs, balance, lookup_k)
        } else {
            None
        };

        let log_w_combined = match (log_w_lookup, log_w_lookup_outputs) {
            (Some(li), Some(lo)) if li.is_finite() && lo.is_finite() => Some(li + lo),
            _ => None,
        };

        let sasamoto_reliable = saddle_reliable(&other_coins, DEFAULT_SADDLE_TAU);

        estimates.push(SubTxEstimate {
            inputs: ins,
            outputs: outs,
            balance,
            log_w_sasamoto,
            log_w_lookup,
            log_w_lookup_outputs,
            log_w_combined,
            count,
            sasamoto_reliable,
        });
    }
    estimates
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
            eprintln!(
                "Sub-tx: {:?} -> {:?}, balance={}, log_w_sas={:?}, log_w_lookup={:?}",
                est.inputs, est.outputs, est.balance, est.log_w_sasamoto, est.log_w_lookup
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
    fn test_sub_tx_uniquely_determined_yields_neg_inf_combined() {
        let tx = fixtures::maurer_fig2();
        let estimates = estimate_sub_txs(&tx, 4);
        assert!(!estimates.is_empty());
        for e in &estimates {
            assert_eq!(e.count, 1, "fig2 sub-txs are uniquely determined");
            if let Some(c) = e.log_w_combined {
                assert!(
                    c == f64::NEG_INFINITY,
                    "expected -inf for uniquely determined sub-tx, got {}",
                    c
                );
            }
        }
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
