//! Subset-sum density estimation primitives: mean log₂ W_signed across coins,
//! plus a 3-tier W(E) estimator (exact DP → lookup → Sasamoto).
//!
//! These are raw measurements. The cost-function framework that consumes them
//! (scaling, thresholding, budget) lives outside this repo.

use crate::validation;
use crate::{
    EmpiricalRegime, SASAMOTO_MIN_N, SignedMethod, Transaction, density_regime, empirical_regime,
    kappa, log_dp_w, log_lookup_w, log_w_for_e_sat, log_w_signed, n_c,
};

#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    pub lookup_k: usize,
    pub force_conservative: bool,
    /// Try exact DP up to this N (default 20 = SASAMOTO_MIN_N collapses the dead band).
    pub exact_threshold: usize,
    /// DP skipped when Σa exceeds this. 10M ≈ a few hundred MB of u128.
    pub dp_max_table: usize,
    /// Sasamoto saddle gate: trust the asymptotic only when `N_c(a)/N < saddle_tau`.
    /// Calibrated against Wasabi2 fixtures: 0.5 keeps small-N/equal-denom batches out
    /// while letting large, broad input sets through. See `empirical-nc` subcommand.
    pub saddle_tau: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            lookup_k: 15,
            force_conservative: false,
            exact_threshold: 20,
            dp_max_table: 10_000_000,
            saddle_tau: 0.5,
        }
    }
}

/// The Sasamoto saddle approximation is trustworthy only when `N ≫ N_c(a)`
/// *and* the ensemble isn't pathologically narrow (equal-denomination inputs
/// have κ_c = 0 at the midpoint per paper eq. 4.3).
pub fn saddle_reliable(a: &[u64], tau: f64) -> bool {
    if a.len() < SASAMOTO_MIN_N {
        return false;
    }
    if matches!(empirical_regime(a), Some(EmpiricalRegime::EqualAmount)) {
        return false;
    }
    n_c(a) / (a.len() as f64) < tau
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorChoice {
    ExactDp,
    Lookup,
    /// `min(Sasamoto, lookup)` — asymptotic regime, min adds conservatism.
    SasamotoLookupMin,
}

impl EstimatorChoice {
    pub fn as_str(&self) -> &'static str {
        match self {
            EstimatorChoice::ExactDp => "exact_dp",
            EstimatorChoice::Lookup => "lookup",
            EstimatorChoice::SasamotoLookupMin => "min(sasamoto, lookup)",
        }
    }
}

fn select_estimator(a: &[u64], config: &EstimatorConfig) -> EstimatorChoice {
    if config.force_conservative {
        return EstimatorChoice::Lookup;
    }
    if a.len() <= config.exact_threshold {
        return EstimatorChoice::ExactDp;
    }
    if saddle_reliable(a, config.saddle_tau) {
        EstimatorChoice::SasamotoLookupMin
    } else {
        EstimatorChoice::Lookup
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    Dense,
    Sparse,
    /// Inputs degenerate or target out of range.
    Unknown,
}

#[derive(Debug, Clone)]
pub struct DensityEstimate {
    pub regime: Regime,
    /// Natural log of W(E).
    pub log_w: Option<f64>,
    pub kappa: Option<f64>,
    pub estimator_used: EstimatorChoice,
    /// True only when backed by a sound lower bound (exact DP or positive lookup).
    /// Sasamoto alone is informative, not a guarantee.
    pub reliable: bool,
}

/// Fail-closed: returns `Sparse`/`Unknown` when no positive sound lower bound exists.
pub fn estimate_density(a: &[u64], e_target: u64, config: &EstimatorConfig) -> DensityEstimate {
    let k = kappa(a);
    let n = a.len();

    if n == 0 || e_target == 0 {
        return DensityEstimate {
            regime: Regime::Unknown,
            log_w: None,
            kappa: k,
            estimator_used: EstimatorChoice::Lookup,
            reliable: false,
        };
    }
    let e_max: u64 = a.iter().sum();
    if e_target >= e_max {
        return DensityEstimate {
            regime: Regime::Unknown,
            log_w: None,
            kappa: k,
            estimator_used: EstimatorChoice::Lookup,
            reliable: false,
        };
    }

    let choice = select_estimator(a, config);

    let (log_w, reliable) = match choice {
        EstimatorChoice::ExactDp => {
            match log_dp_w(a, e_target, config.dp_max_table).filter(|lw| lw.is_finite()) {
                Some(lw) => (Some(lw), true),
                None => {
                    let lookup = log_lookup_w(a, e_target, config.lookup_k);
                    let reliable = matches!(lookup, Some(lw) if lw > 0.0);
                    (lookup, reliable)
                }
            }
        }
        EstimatorChoice::Lookup => {
            let lookup = log_lookup_w(a, e_target, config.lookup_k);
            let reliable = matches!(lookup, Some(lw) if lw > 0.0);
            (lookup, reliable)
        }
        EstimatorChoice::SasamotoLookupMin => {
            let sas = log_w_for_e_sat(a, e_target);
            let lookup = log_lookup_w(a, e_target, config.lookup_k);
            let merged = match (sas, lookup) {
                (Some(s), Some(l)) => {
                    // Sasamoto < 0 = modeling failure; defer to lookup.
                    if s < 0.0 && l >= 0.0 {
                        Some(l)
                    } else {
                        Some(s.min(l))
                    }
                }
                (Some(s), None) => Some(s),
                (None, Some(l)) => Some(l),
                (None, None) => None,
            };
            let reliable = matches!(lookup, Some(l) if l > 0.0);
            (merged, reliable)
        }
    };

    let regime = match log_w {
        Some(lw) if lw > 0.0 => {
            let kappa_dense = density_regime(a, e_target).map(|(_, _, d)| d);
            let confirmed = reliable
                && (matches!(choice, EstimatorChoice::ExactDp) || kappa_dense.unwrap_or(false));
            if confirmed {
                Regime::Dense
            } else {
                Regime::Sparse
            }
        }
        Some(_) => Regime::Sparse,
        None => Regime::Unknown,
    };

    DensityEstimate {
        regime,
        log_w,
        kappa: k,
        estimator_used: choice,
        reliable,
    }
}

/// log₂(W) / N, normalized by coin count for cross-tx comparability. 0 when W ≤ 1.
///
/// ## Which estimator when
///
/// - [`estimate`]: unsigned, one-sided. Asks "how many subsets of `a` sum to `e_target`?"
///   Used for raw density over a single target. Normalized per-coin (log₂W/N).
/// - [`estimate_sub_tx`]: [`estimate`] applied to `(other_inputs, Σ(sub_tx))`. Asks "how many
///   other-input partitions reach this sub-transaction's input sum?"
/// - [`estimate_sub_tx_signed`]: two-sided ±multiset probe. Asks "how many partitions of
///   the *other coins* (inputs minus outputs) reconcile this sub-tx's balance?" Returns
///   nats. This is the per-coin signed-multiset probe.
/// - [`compare_augmented`]: `(before, after, delta)` of mean signed log₂W per coin,
///   comparing the current tx to the augmented tx.
pub fn estimate(a: &[u64], e_target: u64, config: &EstimatorConfig) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    match estimate_density(a, e_target, config).log_w {
        Some(lw) if lw > 0.0 => lw / (n as f64 * 2.0_f64.ln()),
        _ => 0.0,
    }
}

pub fn estimate_sub_tx(
    tx: &Transaction,
    sub_tx_input_indices: &[usize],
    config: &EstimatorConfig,
) -> f64 {
    let sub_tx_sum: u64 = sub_tx_input_indices.iter().map(|&i| tx.inputs[i]).sum();

    let other_inputs: Vec<u64> = tx
        .inputs
        .iter()
        .enumerate()
        .filter(|(i, _)| !sub_tx_input_indices.contains(i))
        .map(|(_, &v)| v)
        .collect();

    estimate(&other_inputs, sub_tx_sum, config)
}

/// Signed ±multiset probe: how many partitions of the other coins match
/// `Σ(my_inputs) - Σ(my_outputs)`? Returns `log W_signed` in nats. The caller
/// picks the estimation method — see [`SignedMethod`].
pub fn estimate_sub_tx_signed(
    tx: &Transaction,
    my_input_indices: &[usize],
    my_output_indices: &[usize],
    lookup_k: usize,
    method: SignedMethod,
) -> Option<f64> {
    let my_in_sum: u64 = my_input_indices.iter().map(|&i| tx.inputs[i]).sum();
    let my_out_sum: u64 = my_output_indices.iter().map(|&i| tx.outputs[i]).sum();
    let sub_balance = my_in_sum as i64 - my_out_sum as i64;

    let other_inputs: Vec<u64> = tx
        .inputs
        .iter()
        .enumerate()
        .filter(|(i, _)| !my_input_indices.contains(i))
        .map(|(_, &v)| v)
        .collect();
    let other_outputs: Vec<u64> = tx
        .outputs
        .iter()
        .enumerate()
        .filter(|(i, _)| !my_output_indices.contains(i))
        .map(|(_, &v)| v)
        .collect();

    log_w_signed(&other_outputs, &other_inputs, sub_balance, lookup_k, method)
}

/// Returns `(before, after, delta)` of mean signed log₂W per coin.
pub fn compare_augmented(
    current_tx: &Transaction,
    new_inputs: &[u64],
    new_outputs: &[u64],
    config: &EstimatorConfig,
    method: SignedMethod,
) -> (f64, f64, f64) {
    let mean_before = tx_mean_signed(current_tx, config, method);

    let mut aug_inputs = current_tx.inputs.clone();
    aug_inputs.extend_from_slice(new_inputs);
    let mut aug_outputs = current_tx.outputs.clone();
    aug_outputs.extend_from_slice(new_outputs);
    let aug_tx = Transaction::new(aug_inputs, aug_outputs);

    let mean_after = tx_mean_signed(&aug_tx, config, method);
    let delta = mean_after - mean_before;

    (mean_before, mean_after, delta)
}

/// Mean log₂ W_signed over all coins. 0 for degenerate txs.
fn tx_mean_signed(tx: &Transaction, config: &EstimatorConfig, method: SignedMethod) -> f64 {
    let total_coins = tx.inputs.len() + tx.outputs.len();
    if total_coins < 2 {
        return 0.0;
    }
    let measurements = validation::per_coin_measurements_fee_aware(tx, config.lookup_k, method);
    let ln2 = std::f64::consts::LN_2;
    let reachable: Vec<f64> = measurements
        .iter()
        .filter_map(|c| c.log_w_signed)
        .map(|v| v / ln2)
        .collect();
    if reachable.is_empty() {
        return 0.0;
    }
    reachable.iter().sum::<f64>() / reachable.len() as f64
}

#[derive(Debug, Clone)]
pub struct RegimeInfo {
    pub kappa: f64,
    pub empirical_regime: Option<EmpiricalRegime>,
    pub estimator: EstimatorChoice,
    pub dense_at_quartile: Option<bool>,
}

pub fn analyze_regime(tx: &Transaction, config: &EstimatorConfig) -> RegimeInfo {
    let k = kappa(&tx.inputs).unwrap_or(f64::NAN);
    let estimator = select_estimator(&tx.inputs, config);

    let dense_at_quartile = {
        let l = *tx.inputs.iter().max().unwrap_or(&0);
        let n = tx.inputs.len() as u64;
        if l == 0 || n == 0 {
            None
        } else {
            let target = n.saturating_mul(l) / 4;
            density_regime(&tx.inputs, target).map(|(_, _, dense)| dense)
        }
    };

    RegimeInfo {
        kappa: k,
        empirical_regime: empirical_regime(&tx.inputs),
        estimator,
        dense_at_quartile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    #[test]
    fn test_estimate_zero_degenerate() {
        let cfg = EstimatorConfig::default();
        assert_eq!(estimate(&[10, 20, 30], 100, &cfg), 0.0);
        assert_eq!(estimate(&[], 10, &cfg), 0.0);
        assert_eq!(estimate(&[10], 0, &cfg), 0.0);
    }

    #[test]
    fn test_estimate_increases_with_n() {
        let config = EstimatorConfig::default();
        let s10 = estimate(&(1..=10).collect::<Vec<u64>>(), 27, &config);
        let s20 = estimate(&(1..=20).collect::<Vec<u64>>(), 105, &config);
        assert!(s20 > s10, "s10={}, s20={}", s10, s20);
    }

    #[test]
    fn test_select_estimator_branches() {
        let mut cfg = EstimatorConfig::default();

        let small: Vec<u64> = (1..=8).collect();
        let edge: Vec<u64> = (1..=20).collect();
        let broad: Vec<u64> = (1..=50).collect();
        let equal = vec![1_000u64; 40];

        assert_eq!(select_estimator(&small, &cfg), EstimatorChoice::ExactDp);
        assert_eq!(select_estimator(&edge, &cfg), EstimatorChoice::ExactDp);

        cfg.exact_threshold = 10;
        let mid: Vec<u64> = (1..=15).collect();
        assert_eq!(select_estimator(&mid, &cfg), EstimatorChoice::Lookup);
        assert_eq!(
            select_estimator(&broad, &cfg),
            EstimatorChoice::SasamotoLookupMin
        );
        // Equal-amount ⇒ saddle unreliable ⇒ Lookup even at large N.
        assert_eq!(select_estimator(&equal, &cfg), EstimatorChoice::Lookup);

        cfg.force_conservative = true;
        assert_eq!(select_estimator(&small, &cfg), EstimatorChoice::Lookup);
        assert_eq!(select_estimator(&broad, &cfg), EstimatorChoice::Lookup);
    }

    #[test]
    fn test_saddle_reliable_gate() {
        let tau = 0.5;
        // Below SASAMOTO_MIN_N: never reliable, regardless of N_c.
        assert!(!saddle_reliable(&(1..=10).collect::<Vec<u64>>(), tau));
        // Broad, large: N_c/N well below 0.5 ⇒ reliable.
        assert!(saddle_reliable(&(1..=50).collect::<Vec<u64>>(), tau));
        // Equal-amount at large N: structurally unreliable.
        assert!(!saddle_reliable(&vec![1_000u64; 40], tau));
    }

    #[test]
    fn test_estimate_small_n_uses_exact_dp() {
        let a: Vec<u64> = (11..=26).collect();
        let cfg = EstimatorConfig::default();
        let target: u64 = a.iter().sum::<u64>() / 2;
        let regime = analyze_regime(&Transaction::new(a.clone(), vec![target]), &cfg);
        assert_eq!(regime.estimator, EstimatorChoice::ExactDp);
        let s = estimate(&a, target, &cfg);
        assert!(s > 0.0);
    }

    #[test]
    fn test_estimate_exact_dp_falls_back_when_table_too_big() {
        let a: Vec<u64> = vec![10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000];
        let cfg = EstimatorConfig {
            exact_threshold: 20,
            dp_max_table: 100,
            lookup_k: 4,
            ..Default::default()
        };
        let s = estimate(&a, 60_000_000, &cfg);
        assert!(s.is_finite());
    }

    #[test]
    fn test_estimate_powers_of_two() {
        let a: Vec<u64> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let s = estimate(&a, 500, &EstimatorConfig::default());
        assert!(s >= 0.0);
    }

    #[test]
    fn test_compare_augmented_positive() {
        let base_tx = Transaction::new(vec![100, 200, 300], vec![150, 250, 200]);
        let config = EstimatorConfig::default();
        let (before, after, _) = compare_augmented(
            &base_tx,
            &[400, 500],
            &[450, 450],
            &config,
            SignedMethod::Lookup,
        );
        assert!(after >= before);
    }

    #[test]
    fn test_estimate_sub_tx() {
        let tx = Transaction::new(vec![10, 20, 30, 40, 50], vec![30, 50, 70]);
        let s = estimate_sub_tx(&tx, &[0, 1], &EstimatorConfig::default());
        assert!(s >= 0.0);
    }

    #[test]
    fn test_regime_info() {
        let tx = fixtures::maurer_fig2();
        let regime = analyze_regime(&tx, &EstimatorConfig::default());
        assert!(regime.kappa > 0.0);
        // Maurer fig 2 is the canonical "no denomination structure" ensemble.
        assert_ne!(
            regime.empirical_regime,
            Some(EmpiricalRegime::RadixGeometric)
        );
    }

    #[test]
    fn test_equal_denominations_regime() {
        let tx = fixtures::equal_denominations();
        let _ = analyze_regime(&tx, &EstimatorConfig::default());
    }

    #[test]
    fn test_incremental_aggregation() {
        let config = EstimatorConfig::default();
        let participants: Vec<(Vec<u64>, Vec<u64>)> = vec![
            (vec![100, 200], vec![150, 150]),
            (vec![300, 50], vec![200, 150]),
            (vec![400, 100], vec![250, 250]),
            (vec![150, 350], vec![300, 200]),
            (vec![500, 200], vec![400, 300]),
        ];

        let mut tx = Transaction::new(participants[0].0.clone(), participants[0].1.clone());
        let mut means = vec![tx_mean_signed(&tx, &config, SignedMethod::Lookup)];
        for (inputs, outputs) in &participants[1..] {
            tx.inputs.extend(inputs);
            tx.outputs.extend(outputs);
            means.push(tx_mean_signed(&tx, &config, SignedMethod::Lookup));
        }

        assert!(*means.last().unwrap() >= means[0]);
    }

    #[test]
    fn test_estimate_sub_tx_signed_balanced() {
        let tx = Transaction::new(vec![100, 200, 300, 400], vec![150, 150, 300, 400]);
        let log_w = estimate_sub_tx_signed(&tx, &[0, 1], &[0, 1], 3, SignedMethod::Lookup);
        assert!(log_w.is_some());
        assert!(log_w.unwrap() >= 0.0);
    }

    #[test]
    fn test_estimate_sub_tx_signed_single_coin() {
        let tx = Transaction::new(vec![10, 20, 30], vec![10, 20, 30]);
        assert!(estimate_sub_tx_signed(&tx, &[0], &[0], 3, SignedMethod::Lookup).is_some());
    }

    #[test]
    fn test_estimate_density_exact_dp_small_n() {
        // W(5000) = C(10, 5) = 252 for 10×1000.
        let a: Vec<u64> = vec![1000; 10];
        let de = estimate_density(&a, 5000, &EstimatorConfig::default());
        assert_eq!(de.estimator_used, EstimatorChoice::ExactDp);
        assert_eq!(de.regime, Regime::Dense);
        assert!(de.reliable);
        assert!((de.log_w.unwrap() - 252f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_density_target_out_of_range() {
        let a: Vec<u64> = vec![10, 20, 30];
        let de = estimate_density(&a, 60, &EstimatorConfig::default());
        assert_eq!(de.regime, Regime::Unknown);
        assert!(de.log_w.is_none());
        assert!(!de.reliable);
    }

    #[test]
    fn test_estimate_density_empty_input() {
        let de = estimate_density(&[], 100, &EstimatorConfig::default());
        assert_eq!(de.regime, Regime::Unknown);
        assert!(!de.reliable);
    }

    #[test]
    fn test_estimate_density_zero_target() {
        let de = estimate_density(&[1, 2, 3], 0, &EstimatorConfig::default());
        assert_eq!(de.regime, Regime::Unknown);
    }

    #[test]
    fn test_estimate_density_equal_denom_dense() {
        let a: Vec<u64> = vec![8; 10];
        let de = estimate_density(&a, 40, &EstimatorConfig::default());
        // N ≤ exact_threshold ⇒ ExactDp (strictly tighter than the old Lookup path).
        assert_eq!(de.estimator_used, EstimatorChoice::ExactDp);
        assert_eq!(de.regime, Regime::Dense);
        assert!(de.reliable);
        assert!(de.log_w.unwrap() > 0.0);
    }

    #[test]
    fn test_estimate_density_large_n_sasamoto_branch() {
        let a: Vec<u64> = (1..=25).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let de = estimate_density(&a, e_mid, &EstimatorConfig::default());
        assert_eq!(de.estimator_used, EstimatorChoice::SasamotoLookupMin);
        assert_eq!(de.regime, Regime::Dense);
        assert!(de.reliable);
    }

    #[test]
    fn test_estimate_density_sparse_when_no_subsets() {
        let a: Vec<u64> = vec![1_000_003, 2_000_029, 3_000_131];
        let de = estimate_density(&a, 1, &EstimatorConfig::default());
        assert_ne!(de.regime, Regime::Dense);
    }

    #[test]
    fn test_estimate_matches_estimate_density() {
        let a: Vec<u64> = (1..=15).collect();
        let e = 60u64;
        let cfg = EstimatorConfig::default();
        let s = estimate(&a, e, &cfg);
        let de = estimate_density(&a, e, &cfg);
        let expected = match de.log_w {
            Some(lw) if lw > 0.0 => lw / (a.len() as f64 * 2.0_f64.ln()),
            _ => 0.0,
        };
        assert!((s - expected).abs() < 1e-12);
    }
}
