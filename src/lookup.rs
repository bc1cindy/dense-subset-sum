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
}
