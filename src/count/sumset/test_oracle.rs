//! Brute-force oracles shared across module-level test suites.

use std::collections::HashMap;

/// Saturating count per sum across all 2^n subsets.
pub(crate) fn subset_sums(set: &[u64]) -> HashMap<u64, u8> {
    let n = set.len();
    let mut out: HashMap<u64, u8> = HashMap::new();
    for mask in 0u32..(1u32 << n) {
        let sum: u64 = (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| set[i])
            .sum();
        out.entry(sum)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);
    }
    out
}

/// Saturating count per sum across subsets of size ≤ `fixed_degree`.
pub(crate) fn subset_sums_bounded(set: &[u64], fixed_degree: usize) -> HashMap<u64, u8> {
    let n = set.len();
    let mut out: HashMap<u64, u8> = HashMap::new();
    for mask in 0u32..(1u32 << n) {
        if mask.count_ones() as usize > fixed_degree {
            continue;
        }
        let sum: u64 = (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| set[i])
            .sum();
        out.entry(sum)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);
    }
    out
}

/// Saturating count per sum across subsets of size exactly `m`.
pub(crate) fn subset_sums_exact(set: &[u64], m: usize) -> HashMap<u64, u8> {
    let n = set.len();
    let mut out: HashMap<u64, u8> = HashMap::new();
    for mask in 0u32..(1u32 << n) {
        if mask.count_ones() as usize != m {
            continue;
        }
        let sum: u64 = (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| set[i])
            .sum();
        out.entry(sum)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);
    }
    out
}

/// Saturating count of (`s_pos`, `s_neg`) pairs with `s_pos` − `s_neg` = target.
pub(crate) fn count_balance(pos: &[u64], neg: &[u64], target: i64) -> u8 {
    let mut total: u8 = 0;
    for p_mask in 0u32..(1u32 << pos.len()) {
        let s_pos: i64 = (0..pos.len())
            .filter(|i| p_mask & (1 << i) != 0)
            .map(|i| i64::try_from(pos[i]).expect("test value fits i64"))
            .sum();
        for n_mask in 0u32..(1u32 << neg.len()) {
            let s_neg: i64 = (0..neg.len())
                .filter(|i| n_mask & (1 << i) != 0)
                .map(|i| i64::try_from(neg[i]).expect("test value fits i64"))
                .sum();
            if s_pos - s_neg == target {
                total = total.saturating_add(1);
            }
        }
    }
    total
}
