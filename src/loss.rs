//! Cost function for co-spend aggregation: mean log₂ W_signed across coins,
//! plus a 3-tier W(E) estimator (exact DP → lookup → Sasamoto).

use crate::validation;
use crate::{
    SASAMOTO_MIN_N, Transaction, arbitrary_distinctness_log2, best_radix_base, coverage_bonus_log2,
    denomination_reward_log2, density_regime, distinguish_coins, is_radix_like_any_base, kappa,
    log_dp_w, log_lookup_w, log_w_for_e_sat,
};

#[derive(Debug, Clone)]
pub struct LossConfig {
    pub lookup_k: usize,
    pub radix_hw_threshold: u32,
    pub force_conservative: bool,
    /// Try exact DP up to this N (default 20 = SASAMOTO_MIN_N collapses the dead band).
    pub exact_threshold: usize,
    /// DP skipped when Σa exceeds this. 10M ≈ a few hundred MB of u128.
    pub dp_max_table: usize,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            lookup_k: 15,
            radix_hw_threshold: 2,
            force_conservative: false,
            exact_threshold: 20,
            dp_max_table: 10_000_000,
        }
    }
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

fn select_estimator(n: usize, use_radix: bool, config: &LossConfig) -> EstimatorChoice {
    if use_radix || config.force_conservative {
        EstimatorChoice::Lookup
    } else if n <= config.exact_threshold {
        EstimatorChoice::ExactDp
    } else if n < SASAMOTO_MIN_N {
        EstimatorChoice::Lookup
    } else {
        EstimatorChoice::SasamotoLookupMin
    }
}

/// Natural-log lower bound from the radix composite (coverage + reward + arb distinctness).
fn radix_composite_log_w(a: &[u64], e_target: u64, hw_threshold: u32) -> Option<f64> {
    let (base, mults) = best_radix_base(a, hw_threshold)?;
    if mults.is_empty() {
        return None;
    }
    let (_, arb) = distinguish_coins(a, base, hw_threshold);
    let cov = coverage_bonus_log2(e_target, base, &mults);
    let rew = denomination_reward_log2(&mults);
    let arb_bonus = arbitrary_distinctness_log2(&arb);
    let log2_w = cov + rew + arb_bonus;
    if log2_w > 0.0 {
        Some(log2_w * 2.0_f64.ln())
    } else {
        None
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
pub fn estimate_density(a: &[u64], e_target: u64, config: &LossConfig) -> DensityEstimate {
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

    let use_radix = is_radix_like_any_base(a, config.radix_hw_threshold);
    let choice = select_estimator(n, use_radix, config);

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
            // Max of two sound lower bounds is the tighter one.
            let merged = if use_radix {
                let radix = radix_composite_log_w(a, e_target, config.radix_hw_threshold);
                match (lookup, radix) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            } else {
                lookup
            };
            let reliable = matches!(merged, Some(lw) if lw > 0.0);
            (merged, reliable)
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
                && (matches!(choice, EstimatorChoice::ExactDp)
                    || use_radix
                    || kappa_dense.unwrap_or(false));
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
/// ## Which score function when
///
/// - [`score`]: unsigned, one-sided. Asks "how many subsets of `a` sum to `e_target`?"
///   Used for raw density over a single target. Normalized per-coin (log₂W/N).
/// - [`score_sub_tx`]: [`score`] applied to `(other_inputs, Σ(sub_tx))`. Asks "how many
///   other-input partitions reach this sub-transaction's input sum?"
/// - [`score_sub_tx_signed`]: two-sided ±multiset probe. Asks "how many partitions of
///   the *other coins* (inputs minus outputs) reconcile this sub-tx's balance?" Returns
///   nats. This is the per-coin privacy probe used by the MVP cost function.
/// - [`marginal_score`]: `(before, after, delta)` mean signed privacy; accept when
///   `delta > 0`. Used for evaluating whether adding inputs/outputs improves privacy.
pub fn score(a: &[u64], e_target: u64, config: &LossConfig) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    match estimate_density(a, e_target, config).log_w {
        Some(lw) if lw > 0.0 => lw / (n as f64 * 2.0_f64.ln()),
        _ => 0.0,
    }
}

pub fn score_sub_tx(tx: &Transaction, sub_tx_input_indices: &[usize], config: &LossConfig) -> f64 {
    let sub_tx_sum: u64 = sub_tx_input_indices.iter().map(|&i| tx.inputs[i]).sum();

    let other_inputs: Vec<u64> = tx
        .inputs
        .iter()
        .enumerate()
        .filter(|(i, _)| !sub_tx_input_indices.contains(i))
        .map(|(_, &v)| v)
        .collect();

    score(&other_inputs, sub_tx_sum, config)
}

/// Signed ±multiset probe: how many partitions of the other coins match
/// `Σ(my_inputs) - Σ(my_outputs)`? Returns `log W_signed` in nats.
pub fn score_sub_tx_signed(
    tx: &Transaction,
    my_input_indices: &[usize],
    my_output_indices: &[usize],
    lookup_k: usize,
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

    crate::log_w_signed_adaptive(&other_outputs, &other_inputs, sub_balance, lookup_k)
}

/// Returns `(before, after, delta)`. Accept when `delta > threshold`.
pub fn marginal_score(
    current_tx: &Transaction,
    new_inputs: &[u64],
    new_outputs: &[u64],
    config: &LossConfig,
) -> (f64, f64, f64) {
    let score_before = tx_signed_privacy(current_tx, config);

    let mut aug_inputs = current_tx.inputs.clone();
    aug_inputs.extend_from_slice(new_inputs);
    let mut aug_outputs = current_tx.outputs.clone();
    aug_outputs.extend_from_slice(new_outputs);
    let aug_tx = Transaction::new(aug_inputs, aug_outputs);

    let score_after = tx_signed_privacy(&aug_tx, config);
    let delta = score_after - score_before;

    (score_before, score_after, delta)
}

/// Mean log₂ W_signed over all coins. 0 for degenerate txs.
fn tx_signed_privacy(tx: &Transaction, config: &LossConfig) -> f64 {
    let total_coins = tx.inputs.len() + tx.outputs.len();
    if total_coins < 2 {
        return 0.0;
    }
    let scores = validation::per_coin_scores_signed_fee_aware(tx, config.lookup_k);
    let ln2 = std::f64::consts::LN_2;
    let reachable: Vec<f64> = scores
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
    pub is_radix: bool,
    pub estimator: EstimatorChoice,
    pub dense_at_quartile: Option<bool>,
}

pub fn analyze_regime(tx: &Transaction, config: &LossConfig) -> RegimeInfo {
    let k = kappa(&tx.inputs).unwrap_or(f64::NAN);
    let is_radix = is_radix_like_any_base(&tx.inputs, config.radix_hw_threshold);
    let estimator = select_estimator(tx.inputs.len(), is_radix, config);

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
        is_radix,
        estimator,
        dense_at_quartile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    #[test]
    fn test_score_zero_degenerate() {
        let cfg = LossConfig::default();
        assert_eq!(score(&[10, 20, 30], 100, &cfg), 0.0);
        assert_eq!(score(&[], 10, &cfg), 0.0);
        assert_eq!(score(&[10], 0, &cfg), 0.0);
    }

    #[test]
    fn test_score_increases_with_n() {
        let config = LossConfig::default();
        let s10 = score(&(1..=10).collect::<Vec<u64>>(), 27, &config);
        let s20 = score(&(1..=20).collect::<Vec<u64>>(), 105, &config);
        assert!(s20 > s10, "s10={}, s20={}", s10, s20);
    }

    #[test]
    fn test_select_estimator_branches() {
        let mut cfg = LossConfig::default();

        assert_eq!(select_estimator(8, false, &cfg), EstimatorChoice::ExactDp);
        assert_eq!(select_estimator(20, false, &cfg), EstimatorChoice::ExactDp);
        cfg.exact_threshold = 10;
        assert_eq!(select_estimator(15, false, &cfg), EstimatorChoice::Lookup);
        assert_eq!(
            select_estimator(50, false, &cfg),
            EstimatorChoice::SasamotoLookupMin
        );
        assert_eq!(select_estimator(100, true, &cfg), EstimatorChoice::Lookup);

        assert_eq!(
            select_estimator(SASAMOTO_MIN_N, false, &cfg),
            EstimatorChoice::SasamotoLookupMin,
        );
        assert_eq!(
            select_estimator(SASAMOTO_MIN_N - 1, false, &cfg),
            EstimatorChoice::Lookup,
        );

        cfg.force_conservative = true;
        assert_eq!(select_estimator(8, false, &cfg), EstimatorChoice::Lookup);
        assert_eq!(select_estimator(50, false, &cfg), EstimatorChoice::Lookup);
    }

    #[test]
    fn test_score_small_n_uses_exact_dp() {
        let a: Vec<u64> = (11..=26).collect();
        let cfg = LossConfig {
            radix_hw_threshold: 1,
            ..LossConfig::default()
        };
        let target: u64 = a.iter().sum::<u64>() / 2;
        let regime = analyze_regime(&Transaction::new(a.clone(), vec![target]), &cfg);
        assert_eq!(regime.estimator, EstimatorChoice::ExactDp);
        let s = score(&a, target, &cfg);
        assert!(s > 0.0);
    }

    #[test]
    fn test_score_exact_dp_falls_back_when_table_too_big() {
        let a: Vec<u64> = vec![10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000];
        let cfg = LossConfig {
            exact_threshold: 20,
            dp_max_table: 100,
            lookup_k: 4,
            ..Default::default()
        };
        let s = score(&a, 60_000_000, &cfg);
        assert!(s.is_finite());
    }

    #[test]
    fn test_score_radix_uses_lookup() {
        let a: Vec<u64> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let config = LossConfig {
            radix_hw_threshold: 1,
            ..Default::default()
        };
        let s = score(&a, 500, &config);
        assert!(s >= 0.0);
    }

    #[test]
    fn test_marginal_score_positive() {
        let base_tx = Transaction::new(vec![100, 200, 300], vec![150, 250, 200]);
        let config = LossConfig::default();
        let (before, after, _) = marginal_score(&base_tx, &[400, 500], &[450, 450], &config);
        assert!(after >= before);
    }

    #[test]
    fn test_score_sub_tx() {
        let tx = Transaction::new(vec![10, 20, 30, 40, 50], vec![30, 50, 70]);
        let s = score_sub_tx(&tx, &[0, 1], &LossConfig::default());
        assert!(s >= 0.0);
    }

    #[test]
    fn test_regime_info() {
        let tx = fixtures::maurer_fig2();
        // threshold=1 excludes values with ≥2 non-zero base-10 digits (default=2 passes them all).
        let config = LossConfig {
            radix_hw_threshold: 1,
            ..LossConfig::default()
        };
        let regime = analyze_regime(&tx, &config);
        assert!(!regime.is_radix);
        assert!(regime.kappa > 0.0);
    }

    #[test]
    fn test_equal_denominations_regime() {
        let tx = fixtures::equal_denominations();
        let _ = analyze_regime(&tx, &LossConfig::default());
    }

    #[test]
    fn test_incremental_aggregation() {
        let config = LossConfig::default();
        let participants: Vec<(Vec<u64>, Vec<u64>)> = vec![
            (vec![100, 200], vec![150, 150]),
            (vec![300, 50], vec![200, 150]),
            (vec![400, 100], vec![250, 250]),
            (vec![150, 350], vec![300, 200]),
            (vec![500, 200], vec![400, 300]),
        ];

        let mut tx = Transaction::new(participants[0].0.clone(), participants[0].1.clone());
        let mut scores = vec![tx_signed_privacy(&tx, &config)];
        for (inputs, outputs) in &participants[1..] {
            tx.inputs.extend(inputs);
            tx.outputs.extend(outputs);
            scores.push(tx_signed_privacy(&tx, &config));
        }

        assert!(*scores.last().unwrap() >= scores[0]);
    }

    #[test]
    fn test_radix_composite_log_w_manual() {
        // 4×8, target=12, base=2: reward=log2(3)+0.5, coverage=4·log2(4)=8.
        let a: Vec<u64> = vec![8, 8, 8, 8];
        let result = radix_composite_log_w(&a, 12, 1).expect("some structure");
        let expected_ln = (3f64.log2() + 0.5 + 8.0) * 2f64.ln();
        assert!((result - expected_ln).abs() < 1e-9);
    }

    #[test]
    fn test_score_sub_tx_signed_balanced() {
        let tx = Transaction::new(vec![100, 200, 300, 400], vec![150, 150, 300, 400]);
        let log_w = score_sub_tx_signed(&tx, &[0, 1], &[0, 1], 3);
        assert!(log_w.is_some());
        assert!(log_w.unwrap() >= 0.0);
    }

    #[test]
    fn test_score_sub_tx_signed_single_coin() {
        let tx = Transaction::new(vec![10, 20, 30], vec![10, 20, 30]);
        assert!(score_sub_tx_signed(&tx, &[0], &[0], 3).is_some());
    }

    #[test]
    fn test_estimate_density_exact_dp_small_n() {
        // W(5000) = C(10, 5) = 252 for 10×1000.
        let a: Vec<u64> = vec![1000; 10];
        let cfg = LossConfig {
            radix_hw_threshold: 0,
            ..Default::default()
        };
        let de = estimate_density(&a, 5000, &cfg);
        assert_eq!(de.estimator_used, EstimatorChoice::ExactDp);
        assert_eq!(de.regime, Regime::Dense);
        assert!(de.reliable);
        assert!((de.log_w.unwrap() - 252f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_density_target_out_of_range() {
        let a: Vec<u64> = vec![10, 20, 30];
        let de = estimate_density(&a, 60, &LossConfig::default());
        assert_eq!(de.regime, Regime::Unknown);
        assert!(de.log_w.is_none());
        assert!(!de.reliable);
    }

    #[test]
    fn test_estimate_density_empty_input() {
        let de = estimate_density(&[], 100, &LossConfig::default());
        assert_eq!(de.regime, Regime::Unknown);
        assert!(!de.reliable);
    }

    #[test]
    fn test_estimate_density_zero_target() {
        let de = estimate_density(&[1, 2, 3], 0, &LossConfig::default());
        assert_eq!(de.regime, Regime::Unknown);
    }

    #[test]
    fn test_estimate_density_radix_path_dense() {
        let a: Vec<u64> = vec![8; 10];
        let cfg = LossConfig {
            radix_hw_threshold: 1,
            ..Default::default()
        };
        let de = estimate_density(&a, 40, &cfg);
        assert_eq!(de.estimator_used, EstimatorChoice::Lookup);
        assert_eq!(de.regime, Regime::Dense);
        assert!(de.reliable);
        assert!(de.log_w.unwrap() > 0.0);
    }

    #[test]
    fn test_estimate_density_large_n_sasamoto_branch() {
        let a: Vec<u64> = (1..=25).collect();
        let cfg = LossConfig {
            radix_hw_threshold: 0,
            ..Default::default()
        };
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let de = estimate_density(&a, e_mid, &cfg);
        assert_eq!(de.estimator_used, EstimatorChoice::SasamotoLookupMin);
        assert_eq!(de.regime, Regime::Dense);
        assert!(de.reliable);
    }

    #[test]
    fn test_estimate_density_sparse_when_no_subsets() {
        let a: Vec<u64> = vec![1_000_003, 2_000_029, 3_000_131];
        let de = estimate_density(&a, 1, &LossConfig::default());
        assert_ne!(de.regime, Regime::Dense);
    }

    #[test]
    fn test_score_matches_estimate_density() {
        let a: Vec<u64> = (1..=15).collect();
        let e = 60u64;
        let cfg = LossConfig::default();
        let s = score(&a, e, &cfg);
        let de = estimate_density(&a, e, &cfg);
        let expected = match de.log_w {
            Some(lw) if lw > 0.0 => lw / (a.len() as f64 * 2.0_f64.ln()),
            _ => 0.0,
        };
        assert!((s - expected).abs() < 1e-12);
    }
}
