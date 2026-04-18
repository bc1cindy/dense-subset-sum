//! Per-coin "unconstrained-ness": W(E=coin_value, A=other coins) via signed probe.

use super::exclude_values;
use crate::{Transaction, log_lookup_w_signed_target_aware, log_w_signed_sasamoto, sumset_cap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoinRole {
    Input,
    Output,
}

impl CoinRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            CoinRole::Input => "in",
            CoinRole::Output => "out",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoinScore {
    pub role: CoinRole,
    pub index: usize,
    pub value: u64,
    pub log_w_signed: Option<f64>,
    pub n_other_coins: usize,
}

/// Zero-fee signed probe: count signed partitions of the other coins balancing `value`.
pub fn per_coin_scores_signed(tx: &Transaction, lookup_k: usize) -> Vec<CoinScore> {
    per_coin_scores_signed_inner(tx, lookup_k, false)
}

/// Input target = v − round(F · v / Σin); outputs unchanged (they don't pay fees).
pub fn per_coin_scores_signed_fee_aware(tx: &Transaction, lookup_k: usize) -> Vec<CoinScore> {
    per_coin_scores_signed_inner(tx, lookup_k, true)
}

/// Above this `positives.len() + negatives.len()`, use Sasamoto (O(N)): the lookup hits
/// the sumset cap and degrades. Twice `crate::SASAMOTO_MIN_N` because this indexes the
/// total ±multiset size, not one side.
const SIGNED_SASAMOTO_THRESHOLD: usize = 40;

fn signed_probe(positives: &[u64], negatives: &[u64], target: i64, lookup_k: usize) -> Option<f64> {
    let n_other = positives.len() + negatives.len();
    if n_other >= SIGNED_SASAMOTO_THRESHOLD
        && let Some(v) = log_w_signed_sasamoto(positives, negatives, target)
        && v.is_finite()
    {
        return Some(v);
    }
    let cap = sumset_cap();
    log_lookup_w_signed_target_aware(positives, negatives, target, lookup_k, cap)
        .filter(|v| v.is_finite())
}

fn per_coin_scores_signed_inner(
    tx: &Transaction,
    lookup_k: usize,
    fee_aware: bool,
) -> Vec<CoinScore> {
    let total = tx.inputs.len() + tx.outputs.len();
    let mut scores = Vec::with_capacity(total);

    let fee = tx.input_sum().saturating_sub(tx.output_sum());
    let input_sum = tx.input_sum();

    for (i, &value) in tx.inputs.iter().enumerate() {
        let other_inputs = exclude_values(&tx.inputs, &[value]);
        let other_outputs = tx.outputs.clone();

        let target = if fee_aware && fee > 0 && input_sum > 0 {
            let fee_share = (fee as f64 * value as f64 / input_sum as f64).round() as i64;
            (value as i64).saturating_sub(fee_share)
        } else {
            value as i64
        };

        let log_w_signed = signed_probe(&other_outputs, &other_inputs, target, lookup_k);
        scores.push(CoinScore {
            role: CoinRole::Input,
            index: i,
            value,
            log_w_signed,
            n_other_coins: other_inputs.len() + other_outputs.len(),
        });
    }

    for (i, &value) in tx.outputs.iter().enumerate() {
        let other_inputs = tx.inputs.clone();
        let other_outputs = exclude_values(&tx.outputs, &[value]);
        let target = value as i64;
        let log_w_signed = signed_probe(&other_inputs, &other_outputs, target, lookup_k);
        scores.push(CoinScore {
            role: CoinRole::Output,
            index: i,
            value,
            log_w_signed,
            n_other_coins: other_inputs.len() + other_outputs.len(),
        });
    }

    scores
}

pub fn print_per_coin_scores(label: &str, tx: &Transaction, scores: &[CoinScore]) {
    println!(
        "\n=== per-coin W scores: {} ({}in/{}out) ===",
        label,
        tx.inputs.len(),
        tx.outputs.len()
    );

    let ln2 = 2.0_f64.ln();
    println!(
        "{:>4} {:>4} {:>15} {:>15}",
        "role", "idx", "value", "log2_w_signed"
    );
    println!("{:─<45}", "");
    for s in scores {
        let sg = s
            .log_w_signed
            .map_or("N/A".into(), |v| format!("{:.3}", v / ln2));
        println!(
            "{:>4} {:>4} {:>15} {:>15}",
            s.role.as_str(),
            s.index,
            s.value,
            sg
        );
    }

    let count_large = |role: CoinRole| {
        scores
            .iter()
            .filter(|s| s.role == role)
            .filter(|s| s.log_w_signed.is_some_and(|lw| lw / ln2 >= 5.0))
            .count()
    };
    let n_in = scores.iter().filter(|s| s.role == CoinRole::Input).count();
    let n_out = scores.iter().filter(|s| s.role == CoinRole::Output).count();
    let large_in = count_large(CoinRole::Input);
    let large_out = count_large(CoinRole::Output);
    println!(
        "  coins with log2_w_signed ≥ 5: {}/{} total  (in {}/{}, out {}/{})",
        large_in + large_out,
        scores.len(),
        large_in,
        n_in,
        large_out,
        n_out
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    #[test]
    fn test_per_coin_scores_cardinality() {
        let tx = fixtures::maurer_fig2();
        let scores = per_coin_scores_signed(&tx, 4);
        assert_eq!(scores.len(), tx.inputs.len() + tx.outputs.len());

        let n_in = scores.iter().filter(|s| s.role == CoinRole::Input).count();
        let n_out = scores.iter().filter(|s| s.role == CoinRole::Output).count();
        assert_eq!(n_in, tx.inputs.len());
        assert_eq!(n_out, tx.outputs.len());

        for s in &scores {
            let expected = match s.role {
                CoinRole::Input => tx.inputs[s.index],
                CoinRole::Output => tx.outputs[s.index],
            };
            assert_eq!(s.value, expected);
            assert_eq!(s.n_other_coins, tx.inputs.len() + tx.outputs.len() - 1);
        }
    }

    #[test]
    fn test_per_coin_scores_wasabi2_small() {
        let txs = fixtures::all_wasabi2_false_cjtxs();
        let (_, tx) = txs
            .iter()
            .find(|(l, _)| *l == "w2_6a6dcc22_17in6out")
            .unwrap();
        let scores = per_coin_scores_signed(tx, 8);

        assert_eq!(scores.len(), 23);

        let ln2 = 2.0_f64.ln();
        let any_nontrivial = scores
            .iter()
            .any(|s| s.log_w_signed.is_some_and(|lw| lw / ln2 >= 1.0));
        assert!(
            any_nontrivial,
            "expected at least one coin with log2_w_signed >= 1"
        );
    }

    /// Signed model invariant: flipping inputs↔outputs must preserve per-coin log_w_signed.
    #[test]
    fn test_per_coin_scores_signed_io_symmetry() {
        let tx1 = Transaction::new(vec![1, 2, 3], vec![6]);
        let tx2 = Transaction::new(vec![6], vec![1, 2, 3]);

        let s1 = per_coin_scores_signed(&tx1, 3);
        let s2 = per_coin_scores_signed(&tx2, 3);
        assert_eq!(s1.len(), s2.len(), "coin counts differ");

        let mut m1: Vec<(u64, Option<f64>)> =
            s1.iter().map(|c| (c.value, c.log_w_signed)).collect();
        let mut m2: Vec<(u64, Option<f64>)> =
            s2.iter().map(|c| (c.value, c.log_w_signed)).collect();
        let cmp = |a: &(u64, Option<f64>), b: &(u64, Option<f64>)| a.0.cmp(&b.0);
        m1.sort_by(cmp);
        m2.sort_by(cmp);

        for ((v1, lw1), (v2, lw2)) in m1.iter().zip(m2.iter()) {
            assert_eq!(v1, v2, "coin values diverged after I/O swap");
            match (lw1, lw2) {
                (Some(a), Some(b)) => assert!(
                    (a - b).abs() < 1e-10,
                    "log_w_signed for v={} diverged: {:.6} vs {:.6}",
                    v1,
                    a,
                    b
                ),
                (None, None) => {}
                other => panic!("one side None, other Some for v={}: {:?}", v1, other),
            }
        }
    }

    #[test]
    fn test_per_coin_scores_signed_balanced_tx_has_nonneg_log() {
        let tx = Transaction::new(vec![5, 7, 11], vec![5, 7, 11]);
        let scores = per_coin_scores_signed(&tx, 3);
        for c in &scores {
            let lw = c
                .log_w_signed
                .unwrap_or_else(|| panic!("coin v={} missing signed score", c.value));
            assert!(
                lw >= 0.0 - 1e-12,
                "coin v={} got log_w_signed={:.4}, expected ≥ 0",
                c.value,
                lw
            );
        }
    }

    #[test]
    fn test_per_coin_scores_signed_fee_aware_adjusts_target() {
        let tx = Transaction::new(vec![100, 200, 300], vec![90, 190, 290]);
        let normal = per_coin_scores_signed(&tx, 3);
        let fee_aware = per_coin_scores_signed_fee_aware(&tx, 3);
        assert_eq!(normal.len(), fee_aware.len());
        for (n, f) in normal.iter().zip(fee_aware.iter()) {
            assert_eq!(n.role, f.role);
            assert_eq!(n.index, f.index);
            assert_eq!(n.value, f.value);
        }
        let has_difference = normal.iter().zip(fee_aware.iter()).any(|(n, f)| {
            match (n.log_w_signed, f.log_w_signed) {
                (Some(a), Some(b)) => (a - b).abs() > 1e-12,
                (None, Some(_)) | (Some(_), None) => true,
                (None, None) => false,
            }
        });
        assert!(
            has_difference,
            "fee-aware scores should differ from non-fee-aware when fee > 0"
        );
    }

    #[test]
    fn test_per_coin_scores_signed_fee_aware_zero_fee_matches() {
        let tx = Transaction::new(vec![10, 20, 30], vec![10, 20, 30]);
        let normal = per_coin_scores_signed(&tx, 3);
        let fee_aware = per_coin_scores_signed_fee_aware(&tx, 3);
        for (n, f) in normal.iter().zip(fee_aware.iter()) {
            assert_eq!(n.log_w_signed, f.log_w_signed);
        }
    }

    #[test]
    fn test_per_coin_scores_signed_large_n_works() {
        let base_values: Vec<u64> = vec![
            5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000,
        ];
        let inputs: Vec<u64> = (0..80)
            .map(|i| base_values[i % base_values.len()] + (i as u64 * 137))
            .collect();
        let total_in: u64 = inputs.iter().sum();
        let fee = 10_000u64;
        let out_total = total_in - fee;
        let mut outputs: Vec<u64> = (0..79)
            .map(|i| base_values[i % base_values.len()] + (i as u64 * 97))
            .collect();
        let partial: u64 = outputs.iter().sum();
        outputs.push(out_total.saturating_sub(partial));
        let tx = Transaction::new(inputs, outputs);

        let scores = per_coin_scores_signed(&tx, 10);
        assert_eq!(scores.len(), 160, "should have 80+80 coin scores");
        let reachable = scores.iter().filter(|c| c.log_w_signed.is_some()).count();
        assert!(
            reachable >= 100,
            "at N=160, Sasamoto path should produce scores for most coins, got {}/160",
            reachable
        );
        let ln2 = std::f64::consts::LN_2;
        let mean_bits: f64 = scores
            .iter()
            .filter_map(|c| c.log_w_signed)
            .map(|v| v / ln2)
            .sum::<f64>()
            / reachable as f64;
        eprintln!(
            "N=160 tx: {}/{} reachable, mean log₂ W_signed = {:.2} bits",
            reachable,
            scores.len(),
            mean_bits
        );
        assert!(
            mean_bits > 0.0,
            "mean bits should be positive for a large diverse tx"
        );
    }

    #[test]
    fn test_per_coin_scores_signed_n50_equal_denoms() {
        let tx = Transaction::new(vec![100_000; 25], vec![100_000; 25]);
        let scores = per_coin_scores_signed(&tx, 10);
        assert_eq!(scores.len(), 50);
        let reachable = scores.iter().filter(|c| c.log_w_signed.is_some()).count();
        assert!(
            reachable >= 40,
            "equal-denom at N=50 should score most coins, got {}/50",
            reachable
        );
    }
}
