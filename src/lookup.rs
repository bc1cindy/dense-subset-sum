//! Exact W(E): brute-force oracle and 0/1 DP for the W(E) tier stack.        

use crate::sasamoto::gcd_slice;

/// Enumerates 2^N subsets. Test oracle; `None` for `N > 25` or when ΣA overflows u64.
pub fn brute_force_w(original_set: &[u64], e_target: u64) -> Option<u128> {
    let n = original_set.len();
    if n > 25 {
        return None;
    }
    original_set
        .iter()
        .copied()
        .try_fold(0u64, u64::checked_add)?;
    let mut count: u128 = 0;
    for mask in 0..(1u64 << n) {
        let sum: u64 = (0..n)
            .filter(|&j| mask & (1 << j) != 0)
            .map(|j| original_set[j])
            .sum();
        if sum == e_target {
            count += 1;
        }
    }
    Some(count)
}

/// Subset-sum DP. `None` for degenerate A, Σa overflow, or table over
/// `max_cells`; `Some(0)` for unreachable E.
pub fn dp_w(original_set: &[u64], e_target: u64, max_cells: usize) -> Option<u128> {
    let NormalizedDp {
        normalized,
        e_norm,
        sum_max,
    } = match normalize_for_dp(original_set, e_target) {
        DpInput::Ready(input) => input,
        DpInput::EarlyZero => return Some(0),
        DpInput::Degenerate => return None,
    };

    let sz = usize::try_from(sum_max).ok()?.checked_add(1)?;
    if sz > max_cells {
        return None;
    }

    let mut dp = vec![0u128; sz];
    dp[0] = 1;

    for &val in &normalized {
        // v == 0: `dp[j] += dp[j]` doubles each cell (zero freely in or out).
        let v = val as usize;
        for j in (v..sz).rev() {
            dp[j] += dp[j - v];
        }
    }

    Some(dp[e_norm as usize])
}

pub fn log_dp_w(original_set: &[u64], e_target: u64, max_cells: usize) -> Option<f64> {
    Some(log_count(dp_w(original_set, e_target, max_cells)?))
}

fn log_count<N: Into<u128>>(w: N) -> f64 {
    let w = w.into();
    if w == 0 {
        f64::NEG_INFINITY
    } else {
        (w as f64).ln()
    }
}

/// Post-gcd DP state. Constructed by [`normalize_for_dp`].
struct NormalizedDp {
    normalized: Vec<u64>,
    e_norm: u64,
    sum_max: u64,
}

/// Outcome of [`normalize_for_dp`].
enum DpInput {
    /// Inputs ready for the DP loop.
    Ready(NormalizedDp),
    /// E unreachable: not a multiple of gcd(A) or > Σa.
    EarlyZero,
    /// A degenerate (empty/all-zero) or Σa overflows u64.
    Degenerate,
}

/// gcd-normalizes A and E for the DP path.
fn normalize_for_dp(set: &[u64], e: u64) -> DpInput {
    let Some(g) = gcd_slice(set) else {
        return DpInput::Degenerate;
    };
    if !e.is_multiple_of(g) {
        return DpInput::EarlyZero;
    }
    let normalized: Vec<u64> = set.iter().map(|&v| v / g).collect();
    let e_norm = e / g;
    let Some(sum_max) = normalized.iter().copied().try_fold(0u64, u64::checked_add) else {
        return DpInput::Degenerate;
    };
    if e_norm > sum_max {
        return DpInput::EarlyZero;
    }
    DpInput::Ready(NormalizedDp {
        normalized,
        e_norm,
        sum_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    #[test]
    fn test_brute_force_w() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9];
        let mut count: u128 = 0;
        for mask in 0..(1u64 << a.len()) {
            let sum: u64 = (0..a.len())
                .filter(|&j| mask & (1 << j) != 0)
                .map(|j| a[j])
                .sum();
            if sum == 15 {
                count += 1;
            }
        }
        assert_eq!(brute_force_w(&a, 15), Some(count));
        assert_eq!(brute_force_w(&a, 0), Some(1));
        assert_eq!(brute_force_w(&a, 35), Some(1));
        assert_eq!(brute_force_w(&a, 36), Some(0));
    }

    #[test]
    fn test_brute_force_w_too_large_returns_none() {
        let a: Vec<u64> = vec![1; 26];
        assert!(brute_force_w(&a, 5).is_none());
    }

    #[test]
    fn test_dp_matches_brute_force() {
        // One 2^N enumeration into a HashMap, then check each E. Cheaper than
        // calling brute_force_w once per E.
        let a: Vec<u64> = (1..=20).collect();
        let n = a.len();

        let mut w_exact: HashMap<u64, u128> = HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry(sum).or_insert(0) += 1;
        }

        for (&e, &w) in &w_exact {
            let dp = dp_w(&a, e, 1_000_000).unwrap();
            assert_eq!(dp, w, "E={}: brute={}, dp={}", e, w, dp);
        }
    }

    #[test]
    fn test_dp_w_gcd_normalization() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        let dp = dp_w(&a, 30, 1_000_000).unwrap();
        let brute = brute_force_w(&a, 30).unwrap();
        assert_eq!(dp, brute);
    }

    #[test]
    fn test_dp_w_e_not_multiple_of_gcd_is_zero() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(dp_w(&a, 15, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_w_e_above_sum_is_zero() {
        let a: Vec<u64> = vec![1, 2, 3];
        let sum: u64 = a.iter().sum();
        assert_eq!(dp_w(&a, sum + 1, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_w_empty_input_is_none() {
        assert!(dp_w(&[], 0, 1_000_000).is_none());
    }

    #[test]
    fn test_dp_w_all_zero_is_none() {
        assert!(dp_w(&[0, 0, 0], 0, 1_000_000).is_none());
    }

    #[test]
    fn test_dp_w_sum_overflow_is_none() {
        let a: Vec<u64> = vec![u64::MAX, 1];
        assert!(dp_w(&a, 0, 1_000_000_000).is_none());
    }

    #[test]
    fn test_dp_w_cap_boundary() {
        let a: Vec<u64> = vec![1, 2, 3];
        let sum: u64 = a.iter().sum();
        let sz = sum as usize + 1;
        assert!(dp_w(&a, sum, sz).is_some());
        assert!(dp_w(&a, sum, sz - 1).is_none());
    }

    #[test]
    fn test_log_dp_w_zero_count_is_neg_inf() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        let lw = log_dp_w(&a, 15, 1_000_000).unwrap();
        assert!(lw.is_infinite() && lw.is_sign_negative());
    }

    #[test]
    fn test_log_dp_w_finite_for_positive_count() {
        let a: Vec<u64> = vec![1, 2, 3, 4];
        let lw = log_dp_w(&a, 5, 1_000_000).unwrap();
        assert!(lw.is_finite() && lw >= 0.0);
    }

    proptest! {
        #[test]
        fn proptest_dp_matches_brute_force(
            set in prop::collection::vec(1u64..=10, 1..=8),
            e in 0u64..=80,
        ) {
            let dp = dp_w(&set, e, 1_000_000).unwrap();
            let brute = brute_force_w(&set, e).unwrap();
            prop_assert_eq!(dp, brute);
        }

        #[test]
        fn proptest_dp_gcd_scale_invariant(
            set in prop::collection::vec(1u64..=10, 1..=8),
            c in 1u64..=10,
            e in 0u64..=80,
        ) {
            let scaled: Vec<u64> = set.iter().map(|&v| v * c).collect();
            let base = dp_w(&set, e, 1_000_000).unwrap();
            let scaled_w = dp_w(&scaled, e * c, 1_000_000).unwrap();
            prop_assert_eq!(base, scaled_w);
        }
    }
}
