//! Log-space `W` aggregations for downstream loss-term consumers.

use crate::count::oracle::dp_w_restricted;
use crate::count::sasamoto::n_c;
use std::fmt;

/// Failure modes for the per-item entries in [`per_input_log_w`] and [`cluster_log_w`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LossError {
    /// `dp_w_restricted` exceeded `dp_max` cells; no count available.
    BudgetExceeded,
    /// All `dp_w_restricted` calls succeeded but the total is 0; `E` is unreachable.
    Unreachable,
}

impl fmt::Display for LossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExceeded => f.write_str("dp_max cells exceeded"),
            Self::Unreachable => f.write_str("E unreachable from the input set"),
        }
    }
}

impl std::error::Error for LossError {}

/// `log(Σ_{M=1..=m_max} W_restricted(other, M, a_j))` per `a_j`. Each entry is
/// [`LossError::BudgetExceeded`] when DP overruns or [`LossError::Unreachable`] when the
/// total over the M range is 0.
#[must_use]
pub fn per_input_log_w(
    other_inputs: &[u64],
    my_inputs: &[u64],
    m_max: usize,
    dp_max: usize,
) -> Vec<Result<f64, LossError>> {
    my_inputs
        .iter()
        .map(|&a_j| sum_restricted_log(other_inputs, 1..=m_max, a_j, dp_max))
        .collect()
}

/// `log W_restricted(other, m_fixed, E)` per `E ∈ my_sumset`. Same error semantics as
/// [`per_input_log_w`].
#[must_use]
pub fn cluster_log_w(
    other_inputs: &[u64],
    my_sumset: &[u64],
    m_fixed: usize,
    dp_max: usize,
) -> Vec<Result<f64, LossError>> {
    my_sumset
        .iter()
        .map(|&e| log_w_restricted(other_inputs, m_fixed, e, dp_max))
        .collect()
}

/// `N_c(other) / |other|`; `None` for empty input or when `n_c` is undefined.
///
/// Returns `Option<f64>` rather than `Result<_, LossError>` because empty input is the
/// only failure mode (`n_c` never errors for non-empty inputs).
#[must_use]
pub fn nc_over_n(other_inputs: &[u64]) -> Option<f64> {
    if other_inputs.is_empty() {
        return None;
    }
    n_c(other_inputs).map(|nc| nc / other_inputs.len() as f64)
}

fn log_w_restricted(a: &[u64], m: usize, e: u64, dp_max: usize) -> Result<f64, LossError> {
    let w = dp_w_restricted(a, m, e, dp_max).map_err(|_| LossError::BudgetExceeded)?;
    if w == 0 {
        Err(LossError::Unreachable)
    } else {
        Ok((w as f64).ln())
    }
}

fn sum_restricted_log(
    a: &[u64],
    m_range: std::ops::RangeInclusive<usize>,
    e: u64,
    dp_max: usize,
) -> Result<f64, LossError> {
    let mut total: u128 = 0;
    for m in m_range {
        let w = dp_w_restricted(a, m, e, dp_max).map_err(|_| LossError::BudgetExceeded)?;
        total = total.checked_add(w).ok_or(LossError::BudgetExceeded)?;
    }
    if total == 0 {
        Err(LossError::Unreachable)
    } else {
        Ok((total as f64).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_input_log_w_counts_replacements() {
        // other = [1,2,3,4], m_max=2. For a_j=5: M=1 → 0; M=2 → {1+4, 2+3} = 2.
        let other = vec![1u64, 2, 3, 4];
        let my = vec![5u64];
        let out = per_input_log_w(&other, &my, 2, 1_000_000);
        let got = out[0].unwrap();
        assert!((got - 2f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn per_input_log_w_unreachable() {
        let other = vec![10u64, 20];
        let my = vec![7u64];
        let out = per_input_log_w(&other, &my, 2, 1_000_000);
        assert_eq!(out[0], Err(LossError::Unreachable));
    }

    #[test]
    fn cluster_log_w_hits_fixed_m() {
        // other = [1,2,3,4,5], m_fixed=2. E=5 → {1+4, 2+3} = 2.
        let other = vec![1u64, 2, 3, 4, 5];
        let out = cluster_log_w(&other, &[5u64], 2, 1_000_000);
        assert!((out[0].unwrap() - 2f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn nc_over_n_is_none_for_empty() {
        assert!(nc_over_n(&[]).is_none());
    }

    #[test]
    fn nc_over_n_matches_formula() {
        let a = vec![100u64, 200, 300, 400];
        let got = nc_over_n(&a).unwrap();
        let expected = n_c(&a).unwrap() / 4.0;
        assert!((got - expected).abs() < 1e-12);
    }
}
