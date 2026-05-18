//! Exact oracles: brute-force enumeration and 0/1 DP. Ground truth for sparse/Sasamoto.
//!
//! ```
//! use dense_subset_sum::{brute_force_w, dp_w};
//!
//! let inputs = vec![1u64, 2, 3, 4, 5];
//! assert_eq!(brute_force_w(&inputs, 5).unwrap(), 3);
//! assert_eq!(dp_w(&inputs, 5, 1_000_000).unwrap(), 3);
//! ```

use crate::count::numeric::gcd_slice;

/// `TooLarge` at `N > 20`: 2²⁰ ≈ 1M subsets runs in milliseconds; larger N falls back to sparse via `Ambiguity::Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BruteError {
    /// `N > 20`: 2^N enumeration too expensive.
    TooLarge,
    /// ΣA overflows u64.
    SumOverflow,
}

impl std::fmt::Display for BruteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge => f.write_str("brute force: N > 20"),
            Self::SumOverflow => f.write_str("brute force: ΣA overflows u64"),
        }
    }
}

impl std::error::Error for BruteError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DpError {
    /// `A` empty or all-zero (gcd undefined).
    EmptyOrAllZero,
    /// ΣA overflows u64.
    SumOverflow,
    /// DP table exceeds `max_cells`.
    ExceedsBudget,
}

impl std::fmt::Display for DpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyOrAllZero => f.write_str("dp: A empty or all zero"),
            Self::SumOverflow => f.write_str("dp: ΣA overflows u64"),
            Self::ExceedsBudget => f.write_str("dp: table size exceeds max_cells"),
        }
    }
}

impl std::error::Error for DpError {}

/// Enumerates 2^N subsets.
///
/// # Errors
///
/// [`BruteError::TooLarge`] when `N > 20`; [`BruteError::SumOverflow`] when ΣA overflows u64.
pub fn brute_force_w(original_set: &[u64], e_target: u64) -> Result<u128, BruteError> {
    let n = original_set.len();
    if n > 20 {
        return Err(BruteError::TooLarge);
    }
    original_set
        .iter()
        .copied()
        .try_fold(0u64, u64::checked_add)
        .ok_or(BruteError::SumOverflow)?;
    let count = (0..(1u64 << n))
        .filter(|&mask| {
            let sum: u64 = (0..n)
                .filter(|&j| mask & (1 << j) != 0)
                .map(|j| original_set[j])
                .sum();
            sum == e_target
        })
        .count() as u128;
    Ok(count)
}

/// Size-restricted brute force; `Ok(0)` when `m > N`.
///
/// # Errors
///
/// [`BruteError::TooLarge`] when `N > 20`.
pub fn brute_force_w_restricted(
    original_set: &[u64],
    m: usize,
    e_target: u64,
) -> Result<u128, BruteError> {
    let n = original_set.len();
    if n > 20 {
        return Err(BruteError::TooLarge);
    }
    if m > n {
        return Ok(0);
    }
    let count = (0..(1u64 << n))
        .filter(|&mask| mask.count_ones() as usize == m)
        .filter(|&mask| {
            let sum: u64 = (0..n)
                .filter(|&j| mask & (1 << j) != 0)
                .map(|j| original_set[j])
                .sum();
            sum == e_target
        })
        .count() as u128;
    Ok(count)
}

/// Subset-sum DP; `Ok(0)` when E unreachable.
///
/// # Errors
///
/// [`DpError::EmptyOrAllZero`], [`DpError::SumOverflow`], [`DpError::ExceedsBudget`].
pub fn dp_w(original_set: &[u64], e_target: u64, max_cells: usize) -> Result<u128, DpError> {
    let NormalizedDp {
        normalized,
        e_norm,
        sum_max,
    } = match normalize_for_dp(original_set, e_target) {
        DpInput::Ready(input) => input,
        DpInput::EarlyZero => return Ok(0),
        DpInput::Empty => return Err(DpError::EmptyOrAllZero),
        DpInput::SumOverflow => return Err(DpError::SumOverflow),
    };

    let sz = sum_max.checked_add(1).ok_or(DpError::SumOverflow)?;
    if sz > max_cells {
        return Err(DpError::ExceedsBudget);
    }

    let mut dp = vec![0u128; sz];
    dp[0] = 1;

    for &v in &normalized {
        // v == 0 doubles each cell (zero freely in or out).
        for j in (v..sz).rev() {
            dp[j] += dp[j - v];
        }
    }

    Ok(dp[e_norm])
}

/// DP restricted to subset size exactly `m`.
///
/// # Errors
///
/// Same as [`dp_w`].
pub fn dp_w_restricted(
    original_set: &[u64],
    m: usize,
    e_target: u64,
    max_cells: usize,
) -> Result<u128, DpError> {
    if m > original_set.len() {
        return Ok(0);
    }
    if m == 0 {
        return Ok(u128::from(e_target == 0));
    }

    let NormalizedDp {
        normalized,
        e_norm,
        sum_max,
    } = match normalize_for_dp(original_set, e_target) {
        DpInput::Ready(input) => input,
        DpInput::EarlyZero => return Ok(0),
        DpInput::Empty => return Err(DpError::EmptyOrAllZero),
        DpInput::SumOverflow => return Err(DpError::SumOverflow),
    };

    let sz = sum_max.checked_add(1).ok_or(DpError::SumOverflow)?;
    let dp_rows = m + 1;
    let cells = dp_rows.checked_mul(sz).ok_or(DpError::ExceedsBudget)?;
    if cells > max_cells {
        return Err(DpError::ExceedsBudget);
    }

    let mut dp = vec![0u128; cells];
    dp[0] = 1;

    for &v in &normalized {
        for mm in (1..=m).rev() {
            let row = mm * sz;
            let prev = (mm - 1) * sz;
            for j in (v..sz).rev() {
                dp[row + j] += dp[prev + j - v];
            }
        }
    }

    Ok(dp[m * sz + e_norm])
}

struct NormalizedDp {
    normalized: Vec<usize>,
    e_norm: usize,
    sum_max: usize,
}

enum DpInput {
    Ready(NormalizedDp),
    /// E not a multiple of gcd(A), or > Σa.
    EarlyZero,
    Empty,
    SumOverflow,
}

fn normalize_for_dp(set: &[u64], e: u64) -> DpInput {
    let Some(g) = gcd_slice(set) else {
        return DpInput::Empty;
    };
    if !e.is_multiple_of(g) {
        return DpInput::EarlyZero;
    }
    let mut normalized: Vec<usize> = Vec::with_capacity(set.len());
    let mut sum_max: usize = 0;
    for &v in set {
        let Ok(v_usize) = usize::try_from(v / g) else {
            return DpInput::SumOverflow;
        };
        let Some(next) = sum_max.checked_add(v_usize) else {
            return DpInput::SumOverflow;
        };
        sum_max = next;
        normalized.push(v_usize);
    }
    let Ok(e_norm) = usize::try_from(e / g) else {
        return DpInput::EarlyZero;
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
    fn brute_force_w_counts_at_target() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9];
        let count = (0..(1u64 << a.len()))
            .filter(|&mask| {
                let sum: u64 = (0..a.len())
                    .filter(|&j| mask & (1 << j) != 0)
                    .map(|j| a[j])
                    .sum();
                sum == 15
            })
            .count() as u128;
        assert_eq!(brute_force_w(&a, 15), Ok(count));
        assert_eq!(brute_force_w(&a, 0), Ok(1));
        assert_eq!(brute_force_w(&a, 35), Ok(1));
        assert_eq!(brute_force_w(&a, 36), Ok(0));
    }

    #[test]
    fn brute_force_w_too_large_returns_none() {
        let a: Vec<u64> = vec![1; 21];
        assert!(brute_force_w(&a, 5).is_err());
    }

    #[test]
    fn brute_force_w_restricted_basic() {
        let a: Vec<u64> = vec![1, 2, 3];
        assert_eq!(brute_force_w_restricted(&a, 0, 0), Ok(1));
        assert_eq!(brute_force_w_restricted(&a, 0, 5), Ok(0));
        assert_eq!(brute_force_w_restricted(&a, 1, 1), Ok(1));
        assert_eq!(brute_force_w_restricted(&a, 1, 4), Ok(0));
        assert_eq!(brute_force_w_restricted(&a, 2, 3), Ok(1));
        assert_eq!(brute_force_w_restricted(&a, 2, 4), Ok(1));
        assert_eq!(brute_force_w_restricted(&a, 3, 6), Ok(1));
        assert_eq!(brute_force_w_restricted(&a, 3, 5), Ok(0));
    }

    #[test]
    fn brute_force_w_restricted_n_too_large_returns_none() {
        assert!(brute_force_w_restricted(&[1u64; 21], 5, 5).is_err());
    }

    #[test]
    fn brute_force_w_restricted_m_above_n_is_zero() {
        assert_eq!(brute_force_w_restricted(&[1u64, 2, 3], 4, 0), Ok(0));
        assert_eq!(brute_force_w_restricted(&[1u64, 2, 3], 100, 0), Ok(0));
    }

    #[test]
    fn brute_force_w_restricted_sums_to_brute_force_w() {
        let a: Vec<u64> = vec![3, 5, 7, 11, 13];
        for e in 0..=40u64 {
            let w_total = brute_force_w(&a, e).unwrap();
            let w_sum: u128 = (0..=a.len())
                .map(|m| brute_force_w_restricted(&a, m, e).unwrap())
                .sum();
            assert_eq!(w_sum, w_total, "e={e}");
        }
    }

    #[test]
    fn dp_matches_brute_force() {
        let a: Vec<u64> = (1..=20).collect();
        let n = a.len();
        let mut w_exact: HashMap<u64, u128> = HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry(sum).or_insert(0) += 1;
        }
        for (&e, &w) in &w_exact {
            let dp = dp_w(&a, e, 1_000_000).unwrap();
            assert_eq!(dp, w, "E={e}: brute={w}, dp={dp}");
        }
    }

    #[test]
    fn dp_w_gcd_normalization() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        let dp = dp_w(&a, 30, 1_000_000).unwrap();
        let brute = brute_force_w(&a, 30).unwrap();
        assert_eq!(dp, brute);
    }

    #[test]
    fn dp_w_e_not_multiple_of_gcd_is_zero() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(dp_w(&a, 15, 1_000_000), Ok(0));
    }

    #[test]
    fn dp_w_e_above_sum_is_zero() {
        let a: Vec<u64> = vec![1, 2, 3];
        let sum: u64 = a.iter().sum();
        assert_eq!(dp_w(&a, sum + 1, 1_000_000), Ok(0));
    }

    #[test]
    fn dp_w_empty_input_returns_none() {
        assert!(dp_w(&[], 0, 1_000_000).is_err());
    }

    #[test]
    fn dp_w_all_zero_returns_none() {
        assert!(dp_w(&[0, 0, 0], 0, 1_000_000).is_err());
    }

    #[test]
    fn dp_w_sum_overflow_returns_none() {
        let a: Vec<u64> = vec![u64::MAX, 1];
        assert!(dp_w(&a, 0, 1_000_000_000).is_err());
    }

    #[test]
    fn dp_w_cap_boundary() {
        let a: Vec<u64> = vec![1, 2, 3];
        let sum: u64 = a.iter().sum();
        let sz = usize::try_from(sum).expect("test sum fits usize") + 1;
        assert!(dp_w(&a, sum, sz).is_ok());
        assert!(dp_w(&a, sum, sz - 1).is_err());
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

    #[test]
    fn dp_restricted_matches_brute_force() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9, 2, 4];
        let sum: u64 = a.iter().sum();
        for m in 0..=a.len() {
            for e in 0..=sum {
                let brute = brute_force_w_restricted(&a, m, e).unwrap();
                let dp = dp_w_restricted(&a, m, e, 1_000_000).unwrap();
                assert_eq!(dp, brute, "m={m}, e={e}: brute={brute}, dp={dp}");
            }
        }
    }

    #[test]
    fn dp_restricted_sum_over_m_matches_dp_w() {
        let a: Vec<u64> = (1..=12).collect();
        let sum: u64 = a.iter().sum();
        for e in [0, 1, 10, sum / 2, sum - 1, sum] {
            let w_total = dp_w(&a, e, 1_000_000).unwrap();
            let w_sum: u128 = (0..=a.len())
                .map(|m| dp_w_restricted(&a, m, e, 1_000_000).unwrap())
                .sum();
            assert_eq!(w_sum, w_total, "e={e}: Σ_m W(m,e)={w_sum}, W(e)={w_total}");
        }
    }

    #[test]
    fn dp_restricted_edges() {
        let a: Vec<u64> = vec![5, 10, 15];
        assert_eq!(dp_w_restricted(&a, 0, 0, 1_000_000), Ok(1));
        assert_eq!(dp_w_restricted(&a, 0, 5, 1_000_000), Ok(0));
        assert_eq!(dp_w_restricted(&a, 4, 10, 1_000_000), Ok(0));
        assert_eq!(dp_w_restricted(&[], 0, 0, 1_000_000), Ok(1));
        assert_eq!(dp_w_restricted(&[], 1, 0, 1_000_000), Ok(0));
    }

    #[test]
    fn dp_restricted_gcd() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(
            dp_w_restricted(&a, 2, 30, 1_000_000).unwrap(),
            brute_force_w_restricted(&a, 2, 30).unwrap(),
        );
        assert_eq!(dp_w_restricted(&a, 2, 15, 1_000_000), Ok(0));
    }

    #[test]
    fn dp_restricted_too_large() {
        let a: Vec<u64> = vec![1, 2, 3];
        assert!(dp_w_restricted(&a, 2, 3, 2).is_err());
    }

    #[test]
    fn dp_w_restricted_sum_overflow_returns_none() {
        let a: Vec<u64> = vec![u64::MAX, 1];
        assert!(dp_w_restricted(&a, 1, 0, 1_000_000_000).is_err());
    }

    proptest! {
        #[test]
        fn proptest_dp_restricted_matches_brute_force(
            set in prop::collection::vec(1u64..=10, 1..=6),
            m in 0usize..=6,
            e in 0u64..=60,
        ) {
            let dp = dp_w_restricted(&set, m, e, 1_000_000).unwrap();
            let brute = brute_force_w_restricted(&set, m, e).unwrap();
            prop_assert_eq!(dp, brute);
        }

        #[test]
        fn proptest_dp_restricted_gcd_scale_invariant(
            set in prop::collection::vec(1u64..=10, 1..=6),
            m in 0usize..=6,
            c in 1u64..=10,
            e in 0u64..=60,
        ) {
            let scaled: Vec<u64> = set.iter().map(|&v| v * c).collect();
            let base = dp_w_restricted(&set, m, e, 1_000_000).unwrap();
            let scaled_w = dp_w_restricted(&scaled, m, e * c, 1_000_000).unwrap();
            prop_assert_eq!(base, scaled_w);
        }
    }
}
