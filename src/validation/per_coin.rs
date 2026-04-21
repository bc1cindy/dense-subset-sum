//! Per-coin "unconstrained-ness": W(E=coin_value, A=other coins) via signed probe.

use super::exclude_values;
use crate::{SignedMethod, Transaction, kappa_c, log_w_signed};

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
pub struct CoinMeasurement {
    pub role: CoinRole,
    pub index: usize,
    pub value: u64,
    pub log_w_signed: Option<f64>,
    pub n_other_coins: usize,
    /// κ_c(x) evaluated at this coin's natural target `x = value / (N_in · max(inputs))`
    /// the critical density that applies if this coin's value were the subset-sum target E.
    /// `None` when x ∉ (0, 1) (e.g. an output larger than N·L).
    pub kappa_c_at_value: Option<f64>,
}

/// Zero-fee signed probe: count signed partitions of the other coins balancing `value`.
pub fn per_coin_measurements(
    tx: &Transaction,
    lookup_k: usize,
    method: SignedMethod,
) -> Vec<CoinMeasurement> {
    per_coin_measurements_inner(tx, lookup_k, method, false)
}

/// Input target = v − round(F · v / Σin); outputs unchanged (they don't pay fees).
pub fn per_coin_measurements_fee_aware(
    tx: &Transaction,
    lookup_k: usize,
    method: SignedMethod,
) -> Vec<CoinMeasurement> {
    per_coin_measurements_inner(tx, lookup_k, method, true)
}

pub fn print_per_coin_measurements(
    label: &str,
    tx: &Transaction,
    measurements: &[CoinMeasurement],
) {
    println!(
        "\n=== per-coin W measurements: {} ({}in/{}out) ===",
        label,
        tx.inputs.len(),
        tx.outputs.len()
    );

    let ln2 = 2.0_f64.ln();
    println!(
        "{:>4} {:>4} {:>15} {:>15} {:>10}",
        "role", "idx", "value", "log2_w_signed", "κ_c"
    );
    println!("{:─<52}", "");
    for s in measurements {
        let sg = s
            .log_w_signed
            .map_or("N/A".into(), |v| format!("{:.3}", v / ln2));
        let kc = s
            .kappa_c_at_value
            .map_or("N/A".into(), |v| format!("{:.4}", v));
        println!(
            "{:>4} {:>4} {:>15} {:>15} {:>10}",
            s.role.as_str(),
            s.index,
            s.value,
            sg,
            kc
        );
    }

    let count_large = |role: CoinRole| {
        measurements
            .iter()
            .filter(|s| s.role == role)
            .filter(|s| s.log_w_signed.is_some_and(|lw| lw / ln2 >= 5.0))
            .count()
    };
    let n_in = measurements
        .iter()
        .filter(|s| s.role == CoinRole::Input)
        .count();
    let n_out = measurements
        .iter()
        .filter(|s| s.role == CoinRole::Output)
        .count();
    let large_in = count_large(CoinRole::Input);
    let large_out = count_large(CoinRole::Output);
    println!(
        "  coins with log2_w_signed ≥ 5: {}/{} total  (in {}/{}, out {}/{})",
        large_in + large_out,
        measurements.len(),
        large_in,
        n_in,
        large_out,
        n_out
    );
}

fn per_coin_measurements_inner(
    tx: &Transaction,
    lookup_k: usize,
    method: SignedMethod,
    fee_aware: bool,
) -> Vec<CoinMeasurement> {
    let total = tx.inputs.len() + tx.outputs.len();
    let mut measurements = Vec::with_capacity(total);

    let fee = tx.input_sum().saturating_sub(tx.output_sum());
    let input_sum = tx.input_sum();

    let n_in = tx.inputs.len() as f64;
    let l_in = tx.inputs.iter().copied().max().unwrap_or(0) as f64;
    let kappa_c_at = |value: u64| -> Option<f64> {
        if n_in <= 0.0 || l_in <= 0.0 {
            return None;
        }
        kappa_c(value as f64 / (n_in * l_in))
    };

    for (i, &value) in tx.inputs.iter().enumerate() {
        let other_inputs = exclude_values(&tx.inputs, &[value]);
        let other_outputs = tx.outputs.clone();

        let target = if fee_aware && fee > 0 && input_sum > 0 {
            let fee_share = (fee as f64 * value as f64 / input_sum as f64).round() as i64;
            (value as i64).saturating_sub(fee_share)
        } else {
            value as i64
        };

        let log_w_signed = log_w_signed(&other_outputs, &other_inputs, target, lookup_k, method);
        measurements.push(CoinMeasurement {
            role: CoinRole::Input,
            index: i,
            value,
            log_w_signed,
            n_other_coins: other_inputs.len() + other_outputs.len(),
            kappa_c_at_value: kappa_c_at(value),
        });
    }

    for (i, &value) in tx.outputs.iter().enumerate() {
        let other_inputs = tx.inputs.clone();
        let other_outputs = exclude_values(&tx.outputs, &[value]);
        let target = value as i64;
        let log_w_signed = log_w_signed(&other_inputs, &other_outputs, target, lookup_k, method);
        measurements.push(CoinMeasurement {
            role: CoinRole::Output,
            index: i,
            value,
            log_w_signed,
            n_other_coins: other_inputs.len() + other_outputs.len(),
            kappa_c_at_value: kappa_c_at(value),
        });
    }

    measurements
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    #[test]
    fn test_per_coin_measurements_cardinality() {
        let tx = fixtures::maurer_fig2();
        let measurements = per_coin_measurements(&tx, 4, SignedMethod::Lookup);
        assert_eq!(measurements.len(), tx.inputs.len() + tx.outputs.len());

        let n_in = measurements
            .iter()
            .filter(|s| s.role == CoinRole::Input)
            .count();
        let n_out = measurements
            .iter()
            .filter(|s| s.role == CoinRole::Output)
            .count();
        assert_eq!(n_in, tx.inputs.len());
        assert_eq!(n_out, tx.outputs.len());

        for s in &measurements {
            let expected = match s.role {
                CoinRole::Input => tx.inputs[s.index],
                CoinRole::Output => tx.outputs[s.index],
            };
            assert_eq!(s.value, expected);
            assert_eq!(s.n_other_coins, tx.inputs.len() + tx.outputs.len() - 1);
        }
    }

    #[test]
    fn test_per_coin_measurements_wasabi2_small() {
        let txs = fixtures::all_wasabi2_false_cjtxs();
        let (_, tx) = txs
            .iter()
            .find(|(l, _)| *l == "w2_6a6dcc22_17in6out")
            .unwrap();
        let measurements = per_coin_measurements(tx, 8, SignedMethod::Lookup);

        assert_eq!(measurements.len(), 23);

        let ln2 = 2.0_f64.ln();
        let any_nontrivial = measurements
            .iter()
            .any(|s| s.log_w_signed.is_some_and(|lw| lw / ln2 >= 1.0));
        assert!(
            any_nontrivial,
            "expected at least one coin with log2_w_signed >= 1"
        );
    }

    /// Signed model invariant: flipping inputs↔outputs must preserve per-coin log_w_signed.
    #[test]
    fn test_per_coin_measurements_io_symmetry() {
        let tx1 = Transaction::new(vec![1, 2, 3], vec![6]);
        let tx2 = Transaction::new(vec![6], vec![1, 2, 3]);

        let s1 = per_coin_measurements(&tx1, 3, SignedMethod::Lookup);
        let s2 = per_coin_measurements(&tx2, 3, SignedMethod::Lookup);
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
    fn test_per_coin_measurements_balanced_tx_has_nonneg_log() {
        let tx = Transaction::new(vec![5, 7, 11], vec![5, 7, 11]);
        let measurements = per_coin_measurements(&tx, 3, SignedMethod::Lookup);
        for c in &measurements {
            let lw = c
                .log_w_signed
                .unwrap_or_else(|| panic!("coin v={} missing signed measurement", c.value));
            assert!(
                lw >= 0.0 - 1e-12,
                "coin v={} got log_w_signed={:.4}, expected ≥ 0",
                c.value,
                lw
            );
        }
    }

    #[test]
    fn test_per_coin_measurements_fee_aware_adjusts_target() {
        let tx = Transaction::new(vec![100, 200, 300], vec![90, 190, 290]);
        let normal = per_coin_measurements(&tx, 3, SignedMethod::Lookup);
        let fee_aware = per_coin_measurements_fee_aware(&tx, 3, SignedMethod::Lookup);
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
            "fee-aware measurements should differ from non-fee-aware when fee > 0"
        );
    }

    #[test]
    fn test_per_coin_measurements_populate_kappa_c_at_value() {
        // N_in=3, L_in=max=3 → x_i = v_i / (N·L) = v_i / 9.
        // Expected: kappa_c(v/9) matches a direct call for every coin (in and out).
        let tx = Transaction::new(vec![1, 2, 3], vec![6]);
        let measurements = per_coin_measurements(&tx, 3, SignedMethod::Lookup);
        let n = tx.inputs.len() as f64;
        let l = *tx.inputs.iter().max().unwrap() as f64;
        for m in &measurements {
            let expected = crate::kappa_c(m.value as f64 / (n * l));
            match (m.kappa_c_at_value, expected) {
                (Some(a), Some(b)) => assert!(
                    (a - b).abs() < 1e-12,
                    "κ_c mismatch for v={}: got {}, expected {}",
                    m.value,
                    a,
                    b
                ),
                (None, None) => {}
                other => panic!("κ_c presence mismatch for v={}: {:?}", m.value, other),
            }
        }
    }

    #[test]
    fn test_per_coin_measurements_fee_aware_zero_fee_matches() {
        let tx = Transaction::new(vec![10, 20, 30], vec![10, 20, 30]);
        let normal = per_coin_measurements(&tx, 3, SignedMethod::Lookup);
        let fee_aware = per_coin_measurements_fee_aware(&tx, 3, SignedMethod::Lookup);
        for (n, f) in normal.iter().zip(fee_aware.iter()) {
            assert_eq!(n.log_w_signed, f.log_w_signed);
        }
    }

    #[test]
    fn test_per_coin_measurements_large_n_works() {
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

        // N=160 is past the lookup cap; exercise the Sasamoto path explicitly.
        let measurements = per_coin_measurements(&tx, 10, SignedMethod::Sasamoto);
        assert_eq!(
            measurements.len(),
            160,
            "should have 80+80 coin measurements"
        );
        let reachable = measurements
            .iter()
            .filter(|c| c.log_w_signed.is_some())
            .count();
        assert!(
            reachable >= 100,
            "at N=160, Sasamoto path should produce measurements for most coins, got {}/160",
            reachable
        );
        let ln2 = std::f64::consts::LN_2;
        let mean_bits: f64 = measurements
            .iter()
            .filter_map(|c| c.log_w_signed)
            .map(|v| v / ln2)
            .sum::<f64>()
            / reachable as f64;
        eprintln!(
            "N=160 tx: {}/{} reachable, mean log₂ W_signed = {:.2} bits",
            reachable,
            measurements.len(),
            mean_bits
        );
        assert!(
            mean_bits > 0.0,
            "mean bits should be positive for a large diverse tx"
        );
    }

    #[test]
    fn test_per_coin_measurements_n50_equal_denoms() {
        let tx = Transaction::new(vec![100_000; 25], vec![100_000; 25]);
        // Exercise the Sasamoto path at N=50 where lookup hits the sumset cap.
        let measurements = per_coin_measurements(&tx, 10, SignedMethod::Sasamoto);
        assert_eq!(measurements.len(), 50);
        let reachable = measurements
            .iter()
            .filter(|c| c.log_w_signed.is_some())
            .count();
        assert!(
            reachable >= 40,
            "equal-denom at N=50 should measure most coins, got {}/50",
            reachable
        );
    }
}
