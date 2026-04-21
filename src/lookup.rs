//! Sumset counts W(E) = #{S ⊆ A : ΣS = E}.
//!
//! - [`brute_force_w`]: exact, N ≲ 25.
//! - [`dp_w`]: exact DP, bails past `max_table_size`.

use crate::sasamoto::gcd_slice;

pub fn brute_force_w(original_set: &[u64], e_target: u64) -> u64 {
    let n = original_set.len();
    assert!(n <= 30, "brute_force_w: N={} too large (max 30)", n);
    let mut count = 0u64;
    for mask in 0..(1u64 << n) {
        let sum: u64 = (0..n)
            .filter(|&j| mask & (1 << j) != 0)
            .map(|j| original_set[j])
            .sum();
        if sum == e_target {
            count += 1;
        }
    }
    count
}

pub fn dp_w(original_set: &[u64], e_target: u64, max_table_size: usize) -> Option<u128> {
    if original_set.is_empty() {
        return None;
    }

    let g = gcd_slice(original_set);
    if g == 0 {
        return None;
    }
    if !e_target.is_multiple_of(g) {
        return Some(0);
    }

    let normalized: Vec<u64> = original_set.iter().map(|&v| v / g).collect();
    let e_norm = e_target / g;
    let sum_max: u64 = normalized.iter().sum();

    if e_norm > sum_max {
        return Some(0);
    }
    if sum_max as usize > max_table_size {
        return None;
    }

    let sz = sum_max as usize + 1;
    let mut dp = vec![0u128; sz];
    dp[0] = 1;

    for &val in &normalized {
        let v = val as usize;
        for j in (v..sz).rev() {
            dp[j] += dp[j - v];
        }
    }

    Some(dp[e_norm as usize])
}

pub fn log_dp_w(original_set: &[u64], e_target: u64, max_table_size: usize) -> Option<f64> {
    let w = dp_w(original_set, e_target, max_table_size)?;
    if w == 0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some((w as f64).ln())
    }
}

/// Exact W(M, E): #subsets of size exactly `m` summing to `e_target`.
///
/// 2D DP: O(N · M · sum_max) time, O(M · sum_max) space. `None` when the table
/// would exceed `max_table_size` cells. Identity: `Σ_{m=0..=N} W(m,E) = W(E)`.
/// Used by the per-input and cluster penalty terms where subset size is bounded.
pub fn dp_w_restricted(
    original_set: &[u64],
    m: usize,
    e_target: u64,
    max_table_size: usize,
) -> Option<u128> {
    if m > original_set.len() {
        return Some(0);
    }
    if m == 0 {
        return Some(if e_target == 0 { 1 } else { 0 });
    }

    let g = gcd_slice(original_set);
    if g == 0 {
        return None;
    }
    if !e_target.is_multiple_of(g) {
        return Some(0);
    }

    let normalized: Vec<u64> = original_set.iter().map(|&v| v / g).collect();
    let e_norm = e_target / g;
    let sum_max: u64 = normalized.iter().sum();

    if e_norm > sum_max {
        return Some(0);
    }

    let sz = sum_max as usize + 1;
    let cells = (m + 1).checked_mul(sz)?;
    if cells > max_table_size {
        return None;
    }

    let mut dp = vec![vec![0u128; sz]; m + 1];
    dp[0][0] = 1;

    for &val in &normalized {
        let v = val as usize;
        for mm in (1..=m).rev() {
            for j in (v..sz).rev() {
                dp[mm][j] += dp[mm - 1][j - v];
            }
        }
    }

    Some(dp[m][e_norm as usize])
}

pub fn log_dp_w_restricted(
    original_set: &[u64],
    m: usize,
    e_target: u64,
    max_table_size: usize,
) -> Option<f64> {
    let w = dp_w_restricted(original_set, m, e_target, max_table_size)?;
    if w == 0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some((w as f64).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_brute_force_w_u64() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9];
        let mut count = 0u64;
        for mask in 0..(1u64 << a.len()) {
            let sum: u64 = (0..a.len())
                .filter(|&j| mask & (1 << j) != 0)
                .map(|j| a[j])
                .sum();
            if sum == 15 {
                count += 1;
            }
        }
        assert_eq!(brute_force_w(&a, 15), count);
        assert_eq!(brute_force_w(&a, 0), 1);
        assert_eq!(brute_force_w(&a, 35), 1);
        assert_eq!(brute_force_w(&a, 36), 0);
    }

    #[test]
    fn test_dp_matches_brute_force() {
        let a: Vec<u64> = (1..=20).collect();
        let n = a.len();

        let mut w_exact: HashMap<u64, u64> = HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry(sum).or_insert(0) += 1;
        }

        for (&e, &w) in &w_exact {
            let dp = dp_w(&a, e, 1_000_000).unwrap();
            assert_eq!(dp, w as u128, "E={}: brute={}, dp={}", e, w, dp);
        }
    }

    #[test]
    fn test_dp_gcd() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(
            dp_w(&a, 30, 1_000_000).unwrap(),
            brute_force_w(&a, 30) as u128
        );
        assert_eq!(dp_w(&a, 15, 1_000_000).unwrap(), 0);
    }

    #[test]
    fn test_dp_too_large() {
        let a: Vec<u64> = vec![1, 2];
        assert!(dp_w(&a, 1, 2).is_none());
    }

    fn brute_force_w_restricted(a: &[u64], m: usize, e: u64) -> u128 {
        let n = a.len();
        let mut count = 0u128;
        for mask in 0..(1u64 << n) {
            if mask.count_ones() as usize != m {
                continue;
            }
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            if sum == e {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn test_dp_restricted_matches_brute_force() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9, 2, 4];
        let sum: u64 = a.iter().sum();
        for m in 0..=a.len() {
            for e in 0..=sum {
                let brute = brute_force_w_restricted(&a, m, e);
                let dp = dp_w_restricted(&a, m, e, 1_000_000).unwrap();
                assert_eq!(dp, brute, "m={}, e={}: brute={}, dp={}", m, e, brute, dp);
            }
        }
    }

    #[test]
    fn test_dp_restricted_sum_over_m_matches_dp_w() {
        // Σ_{m=0..=N} W(m, E) must equal W(E).
        let a: Vec<u64> = (1..=12).collect();
        let sum: u64 = a.iter().sum();
        for e in [0, 1, 10, sum / 2, sum - 1, sum] {
            let w_total = dp_w(&a, e, 1_000_000).unwrap();
            let w_sum: u128 = (0..=a.len())
                .map(|m| dp_w_restricted(&a, m, e, 1_000_000).unwrap())
                .sum();
            assert_eq!(
                w_sum, w_total,
                "e={}: Σ_m W(m,e)={}, W(e)={}",
                e, w_sum, w_total
            );
        }
    }

    #[test]
    fn test_dp_restricted_edges() {
        let a: Vec<u64> = vec![5, 10, 15];
        assert_eq!(dp_w_restricted(&a, 0, 0, 1_000_000), Some(1));
        assert_eq!(dp_w_restricted(&a, 0, 5, 1_000_000), Some(0));
        assert_eq!(dp_w_restricted(&a, 4, 10, 1_000_000), Some(0));
        assert_eq!(dp_w_restricted(&[], 0, 0, 1_000_000), Some(1));
        assert_eq!(dp_w_restricted(&[], 1, 0, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_restricted_gcd() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(
            dp_w_restricted(&a, 2, 30, 1_000_000),
            Some(brute_force_w_restricted(&a, 2, 30))
        );
        assert_eq!(dp_w_restricted(&a, 2, 15, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_restricted_too_large() {
        let a: Vec<u64> = vec![1, 2, 3];
        assert!(dp_w_restricted(&a, 2, 3, 2).is_none());
    }

    #[test]
    fn test_log_dp_restricted_zero_count_is_neg_inf() {
        let a: Vec<u64> = vec![5, 10, 15];
        let lw = log_dp_w_restricted(&a, 2, 3, 1_000_000).unwrap();
        assert!(lw.is_infinite() && lw.is_sign_negative());
    }
}
