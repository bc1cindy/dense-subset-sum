//! Sasamoto saddle-point + signed multiset probe. Cross-validate, never sole signal.
//!
//! ```
//! use dense_subset_sum::log_w_for_e_sat;
//!
//! let inputs: Vec<u64> = (1..=20).collect();
//! let target: u64 = inputs.iter().sum::<u64>() / 2;
//! let log_w = log_w_for_e_sat(&inputs, target);
//! assert!(log_w.is_finite());
//! assert!(log_w > 0.0);
//! ```

use crate::count::density_regime::{Bracket, Regime};
use crate::count::sparse_conv::Field;
use crate::count::sumset::{GradedSumset, GradedSumsetBudget};
use std::fmt;

/// Failure modes for [`log_w_signed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignedError {
    /// Both `plus` and `minus` are empty; the count is undefined.
    EmptyInput,
    /// `count_balance` returned 0; no `(P, N)` pair satisfies the target.
    Unreachable,
}

impl fmt::Display for SignedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("both plus and minus are empty"),
            Self::Unreachable => f.write_str("no balanced (P, N) pair for the target"),
        }
    }
}

impl std::error::Error for SignedError {}

/// `ln W(E)` saddle-point, gated to Dense regime (Sasamoto eq 4.2-4.3). Returns `None`
/// when `set` empty, `e_target ∉ (0, ΣA)`, or [`Bracket::regime`] is not Dense.
#[must_use]
pub fn sasamoto_approx(set: &[u64], e_target: u64) -> Option<f64> {
    if set.is_empty() {
        return None;
    }
    let sum_a: u64 = set.iter().sum();
    if e_target == 0 || e_target >= sum_a {
        return None;
    }
    let bracket = Bracket::new(set.iter().copied(), e_target)?;
    match bracket.regime() {
        Regime::Dense => {}
        Regime::Transitional | Regime::Sparse => return None,
    }
    Some(crate::count::sasamoto::log_w_for_e_sat(set, e_target))
}

/// Size-restricted [`sasamoto_approx`]. Returns `None` when `set` empty, `m ∉ (0, set.len())`,
/// regime not Dense, or `e_target` outside the feasibility interior (Σ smallest m, Σ largest m).
#[must_use]
pub fn sasamoto_approx_m(set: &[u64], m: usize, e_target: u64) -> Option<f64> {
    if set.is_empty() || m == 0 || m >= set.len() {
        return None;
    }
    let bracket = Bracket::new(set.iter().copied(), e_target)?;
    match bracket.regime() {
        Regime::Dense => {}
        Regime::Transitional | Regime::Sparse => return None,
    }
    let mut sorted: Vec<u64> = set.to_vec();
    sorted.sort_unstable();
    let e_min: u64 = sorted[..m].iter().sum();
    let e_max: u64 = sorted[set.len() - m..].iter().sum();
    if e_target <= e_min || e_target >= e_max {
        return None;
    }
    Some(crate::count::sasamoto::log_w_for_m_e_sat(set, m, e_target))
}

/// `ln |{ (P, N) : Σplus − Σminus = target, |P|,|N| ≤ knee }|` via saturating count_balance.
///
/// # Errors
///
/// [`SignedError::EmptyInput`] when both `plus` and `minus` are empty;
/// [`SignedError::Unreachable`] when no balanced pair achieves the target.
pub fn log_w_signed<F: Field>(
    plus: &[u64],
    minus: &[u64],
    target: i64,
    knee: usize,
    budget: GradedSumsetBudget<F>,
) -> Result<f64, SignedError> {
    if plus.is_empty() && minus.is_empty() {
        return Err(SignedError::EmptyInput);
    }
    let p: GradedSumset<F> =
        GradedSumset::builder(plus, budget, &[]).bounded(knee.min(plus.len()).max(0));
    let n: GradedSumset<F> =
        GradedSumset::builder(minus, budget, &[]).bounded(knee.min(minus.len()).max(0));
    let count = p.count_balance(&n, target);
    if count == 0 {
        Err(SignedError::Unreachable)
    } else {
        Ok(f64::from(count).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::count::oracle::brute_force_w;
    use crate::count::sparse_conv::Goldilocks;
    use crate::count::sumset::Bound;

    /// `1.5/√n` tracks asymptotic exactness; 0.30 limit covers finite-N error below ~25.
    fn sasamoto_tolerance(n: usize) -> f64 {
        (1.5 / (n as f64).sqrt()).min(0.30)
    }

    /// Best-case dense gate (κ@max(A) < `κ_c@max(A)`) since Bracket worst-case is too strict at small N.
    #[test]
    fn sasamoto_agrees_with_brute_force_in_dense_regime() {
        use crate::count::sasamoto::log_w_for_e_sat;

        let instances: Vec<Vec<u64>> = vec![
            (1..=20).map(|i| i * 100).collect(),
            (1..=20).map(|i| i * 137).collect(),
        ];
        let mut checked = 0;
        for a in &instances {
            let sum_a: u64 = a.iter().sum();
            for divisor in [8u64, 10, 12, 16] {
                let e = sum_a / divisor;
                if e == 0 || e >= sum_a {
                    continue;
                }
                let bracket =
                    Bracket::new(a.iter().copied(), e).expect("E ∈ (0, sum_a) checked above");
                if bracket.kappa().best() >= bracket.kappa_c().best() {
                    continue;
                }
                let w = brute_force_w(a, e).expect("N≤20 fits brute force");
                if w == 0 {
                    continue;
                }
                let ln_w = (w as f64).ln();
                let log_w_sasa = log_w_for_e_sat(a, e);
                let rel_err = (ln_w - log_w_sasa).abs() / ln_w.abs().max(1.0);
                let tol = sasamoto_tolerance(a.len());
                assert!(
                    rel_err < tol,
                    "Sasamoto disagrees with brute force at E={e} (N={}): \
                     ln(W)={ln_w:.4}, sasamoto={log_w_sasa:.4}, rel_err={rel_err:.4}, tol={tol:.4}",
                    a.len(),
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "No best-case-dense instances exercised; tighten test inputs"
        );
    }

    #[test]
    fn sasamoto_approx_matches_when_dense() {
        use crate::count::density_regime::MAX_MONEY;
        let n: usize = 100;
        let coin = MAX_MONEY / 4 / n as u64;
        let a: Vec<u64> = (0..n).map(|i| coin + (i as u64) * 1000).collect();
        let e: u64 = a.iter().take(50).sum();
        let budget: GradedSumsetBudget = GradedSumsetBudget::default()
            .with_max_size(std::num::NonZeroUsize::new(1_000).unwrap());
        let s: GradedSumset =
            GradedSumset::<Goldilocks>::builder(&a, budget, &[e]).bounded(a.len());
        assert_eq!(s.count_total(e).bound(), Bound::LowerBound);
        let log_w = sasamoto_approx(&a, e).expect("dense regime");
        assert!(log_w.is_finite() && log_w > 0.0);
    }

    #[test]
    fn sasamoto_approx_skips_outside_dense_regime() {
        // Equal-value Whirlpool: kappa_c = 0 forces Sparse regime.
        let a: Vec<u64> = vec![100_000_000; 8];
        let e: u64 = a.iter().take(4).sum();
        assert!(
            sasamoto_approx(&a, e).is_none(),
            "sparse regime must not upgrade"
        );
    }
}
