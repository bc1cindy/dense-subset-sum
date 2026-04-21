//! W-based penalty primitives for CoinJoin cost assembly.
//!
//! Three terms feed the loss function built by downstream callers:
//! - **Per-input** penalty on each `my_input` `a_j`: how many bounded-size
//!   subsets of the other inputs can replace `a_j`? Small ⇒ `a_j` is hard to
//!   hide behind a plausible alternative.
//! - **Cluster** penalty on each `E` reachable by `my_inputs`: how many
//!   fixed-size subsets of the other inputs reach the same balance?
//! - **Global** scalar `N_c(other)/|other|`: Sasamoto reliability proxy.
//!
//! Paper pegs the natural subset-size bound at `⌊√N/2⌋` (Appendix A.7, the
//! N_c saddle). `m_sqrt_over_2` exposes that default.

use crate::lookup::log_dp_w_restricted;
use crate::sasamoto::n_c;

/// Default bounded subset size `⌊√N / 2⌋`, the Sasamoto saddle width.
pub fn m_sqrt_over_2(n: usize) -> usize {
    ((n as f64).sqrt() / 2.0).floor() as usize
}

/// Per-input penalty (nats): for each `a_j ∈ my_inputs`,
/// `log(Σ_{M=1..=m_max} W_restricted(other_inputs, M, a_j))`.
///
/// Returns `f64::NEG_INFINITY` at indices where no subset reaches `a_j`.
/// `None` entries mean the DP table would exceed `dp_max`; caller decides
/// whether to fall back or skip.
pub fn per_input_penalty_log(
    other_inputs: &[u64],
    my_inputs: &[u64],
    m_max: usize,
    dp_max: usize,
) -> Vec<Option<f64>> {
    my_inputs
        .iter()
        .map(|&a_j| sum_restricted_log(other_inputs, 1..=m_max, a_j, dp_max))
        .collect()
}

/// Cluster penalty (nats): for each `E ∈ my_sumset`,
/// `log W_restricted(other_inputs, m_fixed, E)`.
pub fn cluster_penalty_log(
    other_inputs: &[u64],
    my_sumset: &[u64],
    m_fixed: usize,
    dp_max: usize,
) -> Vec<Option<f64>> {
    my_sumset
        .iter()
        .map(|&e| log_dp_w_restricted(other_inputs, m_fixed, e, dp_max))
        .collect()
}

/// Global scalar `N_c(other) / |other|`, the saddle reliability proxy.
/// Returns `NaN` for empty input.
pub fn global_nc_over_n(other_inputs: &[u64]) -> f64 {
    if other_inputs.is_empty() {
        return f64::NAN;
    }
    n_c(other_inputs) / other_inputs.len() as f64
}

fn sum_restricted_log(
    a: &[u64],
    m_range: std::ops::RangeInclusive<usize>,
    e: u64,
    dp_max: usize,
) -> Option<f64> {
    let mut total: u128 = 0;
    for m in m_range {
        let w = crate::lookup::dp_w_restricted(a, m, e, dp_max)?;
        total = total.checked_add(w)?;
    }
    Some(if total == 0 {
        f64::NEG_INFINITY
    } else {
        (total as f64).ln()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_sqrt_over_2_matches_paper_default() {
        assert_eq!(m_sqrt_over_2(16), 2); // √16/2 = 2
        assert_eq!(m_sqrt_over_2(100), 5); // √100/2 = 5
        assert_eq!(m_sqrt_over_2(4), 1);
    }

    #[test]
    fn per_input_penalty_counts_replacements() {
        // other = [1,2,3,4], m_max=2. For a_j=5: M=1 → 0; M=2 → {1+4, 2+3} = 2.
        let other = vec![1u64, 2, 3, 4];
        let my = vec![5u64];
        let out = per_input_penalty_log(&other, &my, 2, 1_000_000);
        let got = out[0].unwrap();
        assert!((got - 2f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn per_input_penalty_unreachable_is_neg_inf() {
        let other = vec![10u64, 20];
        let my = vec![7u64];
        let out = per_input_penalty_log(&other, &my, 2, 1_000_000);
        assert_eq!(out[0], Some(f64::NEG_INFINITY));
    }

    #[test]
    fn cluster_penalty_hits_fixed_m() {
        // other = [1,2,3,4,5], m_fixed=2. E=5 → {1+4, 2+3} = 2.
        let other = vec![1u64, 2, 3, 4, 5];
        let out = cluster_penalty_log(&other, &[5u64], 2, 1_000_000);
        assert!((out[0].unwrap() - 2f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn global_nc_is_nan_for_empty() {
        assert!(global_nc_over_n(&[]).is_nan());
    }

    #[test]
    fn global_nc_matches_formula() {
        let a = vec![100u64, 200, 300, 400];
        let got = global_nc_over_n(&a);
        let expected = n_c(&a) / 4.0;
        assert!((got - expected).abs() < 1e-12);
    }
}
