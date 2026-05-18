//! Radix CoinJoin sumsets, gap analysis, decomposition, and `k × m!` mapping count.
//! Source: "Small Hamming Weight Denominations for CoinJoins".

use std::collections::{BTreeSet, HashMap, HashSet};

/// Exact bitcoin supply ceiling. Distinct from [`crate::MAX_MONEY`] (rounded κ upper bound).
pub const MAX_SATS: u64 = 2_099_999_997_690_000;

pub const DUST_FEERATE_SAT_PER_VBYTE: f64 = 3.0;

pub const EXPECTED_FEERATE_SAT_PER_VBYTE: f64 = 10.0;

/// P2WPKH per <https://bitcoin.stackexchange.com/a/41082>.
pub const P2WPKH_OUTPUT_VBYTES: u64 = 98;

/// `98 × 3.0`; hardcoded because const f64-as-u64 casts are not stable.
pub const DEFAULT_DUST_SATS: u64 = 294;

pub const DEFAULT_MIN_DENOM_SATS: u64 = DEFAULT_DUST_SATS;

pub const DEFAULT_MAX_DENOM_SATS: u64 = 100_000_000;

pub const DEFAULT_MAX_COMBINATION_SIZE: usize = 6;

pub const DEFAULT_MAX_COMBINATION_VALUE_SATS: u64 = DEFAULT_MAX_DENOM_SATS * 2;

/// Excludes unit-gap noise from integer quantization.
pub const DEFAULT_MIN_DIFF: u64 = 2;

pub const DEFAULT_MIN_DIFF_RATIO: f64 = 0.0;

/// Truncates fractional sats (Python `int(...)` semantics).
#[must_use]
pub fn dust_at_feerate(min_vbytes: u64, feerate: f64) -> u64 {
    (min_vbytes as f64 * feerate) as u64
}

/// Nested list where `sumsets[k-1]` is distinct sums in `[1, max_combination_value]`
/// of multisubsets of size `≤ k`. Reuses intermediate results across degrees.
#[must_use]
pub fn radix_sumsets_up_to(
    denoms: &[u64],
    max_k: usize,
    max_combination_value: u64,
) -> Vec<Vec<u64>> {
    if max_k == 0 {
        return Vec::new();
    }
    let mut current: HashSet<u64> = HashSet::from([0u64]);
    let mut accumulated: BTreeSet<u64> = BTreeSet::new();
    let mut out: Vec<Vec<u64>> = Vec::with_capacity(max_k);
    for _ in 1..=max_k {
        let mut next: HashSet<u64> = HashSet::with_capacity(current.len() * denoms.len());
        for &s in &current {
            for &d in denoms {
                if let Some(t) = s.checked_add(d) {
                    if t <= max_combination_value {
                        next.insert(t);
                        if t >= 1 {
                            accumulated.insert(t);
                        }
                    }
                }
            }
        }
        current = next;
        out.push(accumulated.iter().copied().collect());
    }
    out
}

/// Final degree only; for the nested list use [`radix_sumsets_up_to`].
#[must_use]
pub fn radix_sumset(denoms: &[u64], k: usize, max_combination_value: u64) -> Vec<u64> {
    radix_sumsets_up_to(denoms, k, max_combination_value)
        .pop()
        .unwrap_or_default()
}

/// Multiplicity per sum from cwr enumeration; O(C(|denoms|+k, k)).
#[must_use]
pub fn radix_sumset_counts(
    denoms: &[u64],
    k: usize,
    max_combination_value: u64,
) -> HashMap<u64, u64> {
    let mut counts: HashMap<u64, u64> = HashMap::new();
    if k == 0 {
        return counts;
    }
    let mut padded: Vec<u64> = Vec::with_capacity(denoms.len() + 1);
    padded.push(0);
    padded.extend_from_slice(denoms);
    padded.sort_unstable();
    padded.dedup();
    let n = padded.len();
    if n == 0 {
        return counts;
    }
    let mut indices = vec![0usize; k];
    loop {
        let sum = indices
            .iter()
            .try_fold(0u64, |acc, &i| acc.checked_add(padded[i]));
        if let Some(s) = sum {
            if s >= 1 && s <= max_combination_value {
                *counts.entry(s).or_insert(0) += 1;
            }
        }
        let mut pos = k;
        let advanced = loop {
            if pos == 0 {
                break false;
            }
            pos -= 1;
            if indices[pos] + 1 < n {
                indices[pos] += 1;
                for j in (pos + 1)..k {
                    indices[j] = indices[pos];
                }
                break true;
            }
        };
        if !advanced {
            return counts;
        }
    }
}

#[must_use]
pub fn radix_gaps(sumset: &[u64], min_diff: u64) -> Vec<u64> {
    sumset
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d >= min_diff)
        .collect()
}

#[must_use]
pub fn radix_gaps_per_k(sumsets: &[Vec<u64>], min_diff: u64) -> Vec<Vec<u64>> {
    sumsets.iter().map(|s| radix_gaps(s, min_diff)).collect()
}

/// `(b - a) / b` for consecutive pairs, kept iff gap `≥ min_diff` and ratio `≥ min_diff_ratio`.
#[must_use]
pub fn radix_relative_gaps(sumset: &[u64], min_diff: u64, min_diff_ratio: f64) -> Vec<f64> {
    sumset
        .windows(2)
        .filter_map(|w| {
            let (a, b) = (w[0], w[1]);
            let d = b - a;
            if d < min_diff || b == 0 {
                return None;
            }
            let p = d as f64 / b as f64;
            (p >= min_diff_ratio).then_some(p)
        })
        .collect()
}

#[must_use]
pub fn radix_relative_gaps_per_k(
    sumsets: &[Vec<u64>],
    min_diff: u64,
    min_diff_ratio: f64,
) -> Vec<Vec<f64>> {
    sumsets
        .iter()
        .map(|s| radix_relative_gaps(s, min_diff, min_diff_ratio))
        .collect()
}

/// `(bin_start, count / bin_width)` over `[sumset[0], sumset.last()]`. Density 1 means dense.
#[must_use]
pub fn sumset_density(sumset: &[u64], bin_width: u64) -> Vec<(u64, f64)> {
    if sumset.is_empty() || bin_width == 0 {
        return Vec::new();
    }
    let lo = sumset[0];
    let hi = *sumset.last().unwrap();
    let mut out = Vec::new();
    let mut bin_start = lo;
    let mut idx = 0usize;
    while bin_start <= hi {
        let bin_end = bin_start.saturating_add(bin_width - 1).min(hi);
        while idx < sumset.len() && sumset[idx] < bin_start {
            idx += 1;
        }
        let mut count = 0u64;
        let mut j = idx;
        while j < sumset.len() && sumset[j] <= bin_end {
            count += 1;
            j += 1;
        }
        let width = (bin_end - bin_start + 1) as f64;
        out.push((bin_start, count as f64 / width));
        bin_start = match bin_start.checked_add(bin_width) {
            Some(n) => n,
            None => break,
        };
    }
    out
}

/// Largest element of sorted `sumset` that is `≤ target`.
#[must_use]
pub fn approximate_from_below(sumset: &[u64], target: u64) -> Option<u64> {
    let idx = sumset.partition_point(|&s| s <= target);
    if idx == 0 {
        None
    } else {
        Some(sumset[idx - 1])
    }
}

/// Ascending multiset of `≤ max_k` denoms summing to `target`. DFS, first hit (not min-k).
///
/// ```
/// use dense_subset_sum::radix_decompose;
/// assert_eq!(radix_decompose(&[1u64, 2, 5], 8, 3), Some(vec![1, 2, 5]));
/// assert_eq!(radix_decompose(&[3u64, 5], 1, 10), None);
/// ```
#[must_use]
pub fn radix_decompose(denoms: &[u64], target: u64, max_k: usize) -> Option<Vec<u64>> {
    if target == 0 {
        return Some(Vec::new());
    }
    if denoms.is_empty() || max_k == 0 {
        return None;
    }
    let mut sorted: Vec<u64> = denoms.iter().copied().filter(|&d| d > 0).collect();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.is_empty() {
        return None;
    }
    let mut out: Vec<u64> = Vec::with_capacity(max_k);
    let max_idx = sorted.len() - 1;
    if try_decompose(&sorted, target, max_idx, max_k, &mut out) {
        out.sort_unstable();
        Some(out)
    } else {
        None
    }
}

/// `max_idx` enforces non-increasing picks: avoids permutation duplicates.
fn try_decompose(
    denoms: &[u64],
    remaining: u64,
    max_idx: usize,
    depth_left: usize,
    out: &mut Vec<u64>,
) -> bool {
    if remaining == 0 {
        return true;
    }
    if depth_left == 0 {
        return false;
    }
    for i in (0..=max_idx).rev() {
        let d = denoms[i];
        if d > remaining {
            continue;
        }
        out.push(d);
        if try_decompose(denoms, remaining - d, i, depth_left - 1, out) {
            return true;
        }
        out.pop();
    }
    false
}

/// `n!` as `u128`; `None` for `n ≥ 35` (overflow).
#[must_use]
pub fn factorial(n: usize) -> Option<u128> {
    let mut r: u128 = 1;
    for i in 2..=n {
        r = r.checked_mul(u128::try_from(i).ok()?)?;
    }
    Some(r)
}

/// `k × m!` equivalent mappings for a `k:1` exchange (sub-transaction model).
#[must_use]
pub fn radix_mapping_count(k: usize, m: usize) -> Option<u128> {
    let k_u = u128::try_from(k).ok()?;
    k_u.checked_mul(factorial(m)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::count::denoms::standard_denoms_in_range;

    #[test]
    fn radix_sumset_matches_notebook_125_k3() {
        // Hand-computed reference from combinations_with_replacement({0,1,2,5}, 3):
        // distinct sums in [1, ∞) are {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15}.
        let denoms = vec![1u64, 2, 5];
        let sumset = radix_sumset(&denoms, 3, 100);
        assert_eq!(sumset, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]);
    }

    #[test]
    fn radix_sumset_zero_k_is_empty() {
        assert!(radix_sumset(&[1, 2, 5], 0, 100).is_empty());
    }

    #[test]
    fn radix_sumset_respects_max_combination_value() {
        let sumset = radix_sumset(&[1u64, 2, 5], 3, 5);
        assert_eq!(sumset, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn radix_sumset_k1_equals_denoms() {
        let denoms = vec![3u64, 7, 11];
        let sumset = radix_sumset(&denoms, 1, 100);
        assert_eq!(sumset, vec![3, 7, 11]);
    }

    #[test]
    fn radix_sumset_includes_smaller_sizes() {
        // With k=2 and {3, 5}: size-1 gives {3, 5}, size-2 gives {6, 8, 10}.
        // Union sorted: {3, 5, 6, 8, 10}.
        let sumset = radix_sumset(&[3u64, 5], 2, 100);
        assert_eq!(sumset, vec![3, 5, 6, 8, 10]);
    }

    #[test]
    fn radix_gaps_filters_below_min_diff() {
        let sumset = vec![1u64, 2, 5, 6, 100];
        assert_eq!(radix_gaps(&sumset, 2), vec![3, 94]);
    }

    #[test]
    fn radix_gaps_keeps_unit_when_min_diff_one() {
        let sumset = vec![1u64, 2, 5, 6, 100];
        assert_eq!(radix_gaps(&sumset, 1), vec![1, 3, 1, 94]);
    }

    #[test]
    fn radix_gaps_empty_when_sumset_singleton() {
        assert!(radix_gaps(&[42u64], 1).is_empty());
        assert!(radix_gaps(&[], 1).is_empty());
    }

    #[test]
    fn radix_gaps_per_k_matches_per_call_map() {
        let sumsets = radix_sumsets_up_to(&[1u64, 2, 5], 3, 50);
        let batched = radix_gaps_per_k(&sumsets, 2);
        let manual: Vec<Vec<u64>> = sumsets.iter().map(|s| radix_gaps(s, 2)).collect();
        assert_eq!(batched, manual);
        assert_eq!(batched.len(), 3);
    }

    #[test]
    fn radix_relative_gaps_per_k_matches_per_call_map() {
        let sumsets = radix_sumsets_up_to(&[1u64, 2, 5], 3, 50);
        let batched = radix_relative_gaps_per_k(&sumsets, 1, 0.0);
        let manual: Vec<Vec<f64>> = sumsets
            .iter()
            .map(|s| radix_relative_gaps(s, 1, 0.0))
            .collect();
        assert_eq!(batched, manual);
    }

    #[test]
    fn radix_decompose_simple_125() {
        let d = radix_decompose(&[1u64, 2, 5], 8, 3).unwrap();
        assert_eq!(d.iter().sum::<u64>(), 8);
        assert!(d.len() <= 3);
        assert_eq!(d, vec![1, 2, 5]);
    }

    #[test]
    fn radix_decompose_requires_backtrack() {
        // {3, 5, 7}, target = 9: greedy picks 7 → remaining 2 (dead end);
        // backtracks to 5 → remaining 4 → 3 → remaining 1 (dead end);
        // backtracks to 3 → 3 → 3 ✓.
        let d = radix_decompose(&[3u64, 5, 7], 9, 3).unwrap();
        assert_eq!(d.iter().sum::<u64>(), 9);
        assert!(d.len() <= 3);
        assert_eq!(d, vec![3, 3, 3]);
    }

    #[test]
    fn radix_decompose_zero_target_returns_empty() {
        assert_eq!(radix_decompose(&[1, 2, 5], 0, 3), Some(vec![]));
    }

    #[test]
    fn radix_decompose_target_beyond_capacity_is_none() {
        // {1, 2, 5}, max_k = 3: largest reachable sum is 5+5+5 = 15. 100 unreachable.
        assert_eq!(radix_decompose(&[1u64, 2, 5], 100, 3), None);
    }

    #[test]
    fn radix_decompose_unreachable_is_none() {
        // {3, 5}, target = 1: no decomposition possible at any k.
        assert_eq!(radix_decompose(&[3u64, 5], 1, 10), None);
    }

    #[test]
    fn radix_decompose_max_k_zero_with_nonzero_target_is_none() {
        assert_eq!(radix_decompose(&[1, 2, 5], 5, 0), None);
    }

    #[test]
    fn radix_decompose_empty_denoms_is_none() {
        assert_eq!(radix_decompose(&[], 5, 3), None);
    }

    #[test]
    fn radix_decompose_handles_repeats_in_input() {
        // Duplicated input denoms should not change behavior (internal dedup).
        let d = radix_decompose(&[1u64, 1, 2, 2, 5], 8, 3).unwrap();
        assert_eq!(d, vec![1, 2, 5]);
    }

    #[test]
    fn radix_decompose_skips_zero_denoms() {
        // 0 denoms would loop infinitely; must be filtered out internally.
        let d = radix_decompose(&[0u64, 1, 2, 5], 8, 3).unwrap();
        assert_eq!(d, vec![1, 2, 5]);
    }

    #[test]
    fn radix_decompose_consistent_with_sumset() {
        // Every sum in radix_sumset must be decomposable via radix_decompose.
        let denoms = vec![1u64, 2, 5];
        let max_k = 4;
        let sumset = radix_sumset(&denoms, max_k, 50);
        for &target in &sumset {
            let d = radix_decompose(&denoms, target, max_k);
            assert!(
                d.is_some(),
                "target {target} should have decomposition ≤ {max_k}"
            );
            let d = d.unwrap();
            assert_eq!(d.iter().sum::<u64>(), target, "target={target}");
            assert!(
                d.len() <= max_k,
                "target={target} used {} > {max_k}",
                d.len()
            );
            for x in &d {
                assert!(denoms.contains(x), "{x} not in denoms");
            }
        }
    }

    #[test]
    fn radix_decompose_canonical_ascending_order() {
        let d = radix_decompose(&[1u64, 2, 5, 10], 18, 4).unwrap();
        for w in d.windows(2) {
            assert!(w[0] <= w[1], "{:?} not ascending", d);
        }
    }

    #[test]
    fn radix_decompose_enables_radix_mapping_count_pipeline() {
        // End-to-end: detect → decompose → count.
        let denoms = standard_denoms_in_range(1, 100);
        let target = 18u64;
        let decomp = radix_decompose(&denoms, target, 6).unwrap();
        let k_distinct = decomp.iter().collect::<HashSet<_>>().len();
        // For multiplicity m=1 (each denom appears once in this tiny example),
        // k × m! = k × 1 = k_distinct.
        let mappings = radix_mapping_count(k_distinct, 1).unwrap();
        assert_eq!(mappings as usize, k_distinct);
    }

    #[test]
    fn approximate_from_below_finds_largest_le() {
        let sumset = vec![1u64, 3, 7, 10, 15];
        assert_eq!(approximate_from_below(&sumset, 0), None);
        assert_eq!(approximate_from_below(&sumset, 1), Some(1));
        assert_eq!(approximate_from_below(&sumset, 6), Some(3));
        assert_eq!(approximate_from_below(&sumset, 10), Some(10));
        assert_eq!(approximate_from_below(&sumset, 100), Some(15));
    }

    #[test]
    fn approximate_from_below_empty_sumset_is_none() {
        assert_eq!(approximate_from_below(&[], 100), None);
    }

    #[test]
    fn radix_mapping_count_k_times_m_factorial() {
        // k × m! for the (k:1) exchange in the sub-tx model.
        assert_eq!(radix_mapping_count(1, 0), Some(1)); // 1 × 0! = 1
        assert_eq!(radix_mapping_count(1, 1), Some(1)); // 1 × 1! = 1
        assert_eq!(radix_mapping_count(3, 2), Some(6)); // 3 × 2! = 6
        assert_eq!(radix_mapping_count(3, 5), Some(360)); // 3 × 120 = 360
        assert_eq!(radix_mapping_count(0, 4), Some(0)); // 0 × 4! = 0
    }

    #[test]
    fn radix_mapping_count_overflow_returns_none() {
        // 35! > u128::MAX (~3.4e38); 34! ≈ 2.95e38 fits.
        assert!(radix_mapping_count(1, 34).is_some());
        assert!(radix_mapping_count(1, 35).is_none());
    }

    #[test]
    fn factorial_small_values() {
        assert_eq!(factorial(0), Some(1));
        assert_eq!(factorial(1), Some(1));
        assert_eq!(factorial(2), Some(2));
        assert_eq!(factorial(5), Some(120));
        assert_eq!(factorial(10), Some(3_628_800));
    }

    #[test]
    fn factorial_overflow_at_35() {
        assert!(factorial(34).is_some());
        assert!(factorial(35).is_none());
    }

    #[test]
    fn factorial_consistent_with_radix_mapping_count_at_k1() {
        // k=1 → k × m! = m!, so radix_mapping_count(1, m) = factorial(m).
        for m in 0..=20 {
            assert_eq!(radix_mapping_count(1, m), factorial(m), "m={m}");
        }
    }

    #[test]
    fn radix_sumsets_up_to_matches_per_k_calls() {
        let denoms = vec![1u64, 2, 5];
        let all = radix_sumsets_up_to(&denoms, 4, 100);
        assert_eq!(all.len(), 4);
        for k in 1..=4 {
            assert_eq!(all[k - 1], radix_sumset(&denoms, k, 100), "k={k}");
        }
    }

    #[test]
    fn radix_sumsets_up_to_is_nested() {
        // sumsets[k-1] ⊆ sumsets[k] (size ≤ k+1 includes size ≤ k cases).
        let denoms = vec![3u64, 7, 11];
        let all = radix_sumsets_up_to(&denoms, 5, 200);
        for k in 0..(all.len() - 1) {
            let smaller: HashSet<u64> = all[k].iter().copied().collect();
            let larger: HashSet<u64> = all[k + 1].iter().copied().collect();
            assert!(smaller.is_subset(&larger), "k={k} not nested");
        }
    }

    #[test]
    fn radix_sumsets_up_to_zero_is_empty() {
        assert!(radix_sumsets_up_to(&[1, 2, 5], 0, 100).is_empty());
    }

    #[test]
    fn radix_sumset_counts_matches_notebook_125_k2() {
        // cwr({0,1,2,5}, 2) sums with multiplicity (after dedup of {0,1,2,5}):
        // (0,0)→0 excl, (0,1)→1, (0,2)→2, (0,5)→5, (1,1)→2, (1,2)→3, (1,5)→6,
        // (2,2)→4, (2,5)→7, (5,5)→10. Distinct sums: 1,2,3,4,5,6,7,10 with
        // count[2] = 2 (two ways).
        let denoms = vec![1u64, 2, 5];
        let counts = radix_sumset_counts(&denoms, 2, 100);
        assert_eq!(counts.get(&1), Some(&1));
        assert_eq!(counts.get(&2), Some(&2));
        assert_eq!(counts.get(&3), Some(&1));
        assert_eq!(counts.get(&4), Some(&1));
        assert_eq!(counts.get(&5), Some(&1));
        assert_eq!(counts.get(&6), Some(&1));
        assert_eq!(counts.get(&7), Some(&1));
        assert_eq!(counts.get(&10), Some(&1));
        assert_eq!(counts.len(), 8);
    }

    #[test]
    fn radix_sumset_counts_keys_match_radix_sumset() {
        // Keys of counts must equal radix_sumset (modulo set vs vec).
        let denoms = vec![3u64, 5, 7];
        for k in 1..=3 {
            let counts = radix_sumset_counts(&denoms, k, 50);
            let sumset: HashSet<u64> = radix_sumset(&denoms, k, 50).into_iter().collect();
            let keys: HashSet<u64> = counts.keys().copied().collect();
            assert_eq!(keys, sumset, "k={k}");
        }
    }

    #[test]
    fn radix_sumset_counts_zero_k_is_empty() {
        assert!(radix_sumset_counts(&[1, 2, 5], 0, 100).is_empty());
    }

    #[test]
    fn radix_sumset_counts_respects_max_combination_value() {
        let counts = radix_sumset_counts(&[1u64, 2, 5], 2, 4);
        for &s in counts.keys() {
            assert!((1..=4).contains(&s), "sum {s} outside [1, 4]");
        }
    }

    #[test]
    fn notebook_constants_match_spec() {
        assert_eq!(MAX_SATS, 2_099_999_997_690_000);
        assert_eq!(DUST_FEERATE_SAT_PER_VBYTE, 3.0);
        assert_eq!(EXPECTED_FEERATE_SAT_PER_VBYTE, 10.0);
        assert_eq!(P2WPKH_OUTPUT_VBYTES, 98);
        assert_eq!(DEFAULT_DUST_SATS, 294);
        assert_eq!(DEFAULT_MIN_DENOM_SATS, 294);
        assert_eq!(DEFAULT_MAX_DENOM_SATS, 100_000_000);
        assert_eq!(DEFAULT_MAX_COMBINATION_SIZE, 6);
        assert_eq!(DEFAULT_MAX_COMBINATION_VALUE_SATS, 200_000_000);
        assert_eq!(DEFAULT_MIN_DIFF, 2);
        assert_eq!(DEFAULT_MIN_DIFF_RATIO, 0.0);
        assert_eq!(
            DEFAULT_DUST_SATS,
            dust_at_feerate(P2WPKH_OUTPUT_VBYTES, DUST_FEERATE_SAT_PER_VBYTE)
        );
    }

    #[test]
    fn radix_relative_gaps_matches_notebook_formula() {
        // sumset = [1, 3, 7, 10, 15]; pairs (a,b) → (b-a)/b
        let sumset = vec![1u64, 3, 7, 10, 15];
        let rel = radix_relative_gaps(&sumset, 1, 0.0);
        assert_eq!(rel.len(), 4);
        assert!((rel[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((rel[1] - 4.0 / 7.0).abs() < 1e-12);
        assert!((rel[2] - 3.0 / 10.0).abs() < 1e-12);
        assert!((rel[3] - 5.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn radix_relative_gaps_filters_min_diff() {
        // (1,3)d=2, (3,7)d=4, (7,10)d=3, (10,15)d=5 → min_diff=4 keeps only d=4 and d=5
        let sumset = vec![1u64, 3, 7, 10, 15];
        let rel = radix_relative_gaps(&sumset, 4, 0.0);
        assert_eq!(rel.len(), 2);
    }

    #[test]
    fn radix_relative_gaps_filters_min_ratio() {
        // ratios: 0.667, 0.571, 0.300, 0.333 → min_ratio=0.5 keeps two
        let sumset = vec![1u64, 3, 7, 10, 15];
        let rel = radix_relative_gaps(&sumset, 1, 0.5);
        assert_eq!(rel.len(), 2);
    }

    #[test]
    fn radix_relative_gaps_empty_when_short() {
        assert!(radix_relative_gaps(&[42u64], 1, 0.0).is_empty());
        assert!(radix_relative_gaps(&[], 1, 0.0).is_empty());
    }

    #[test]
    fn dust_at_feerate_truncates_like_python_int() {
        // Python int(98 * 3.0) = 294, int(98 * 3.5) = 343, int(98 * 1.0) = 98.
        assert_eq!(dust_at_feerate(98, 3.0), 294);
        assert_eq!(dust_at_feerate(98, 3.5), 343);
        assert_eq!(dust_at_feerate(98, 1.0), 98);
        // Fractional truncation: int(98 * 2.999) = 293
        assert_eq!(dust_at_feerate(98, 2.999), 293);
    }

    #[test]
    fn sumset_density_uniform_sumset() {
        // sumset = [10, 11, 12, ..., 19] dense in [10, 19].
        let sumset: Vec<u64> = (10..=19).collect();
        let d = sumset_density(&sumset, 5);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0], (10, 1.0)); // bin [10, 14] has 5 elems / 5 width = 1.0
        assert_eq!(d[1], (15, 1.0)); // bin [15, 19] has 5 elems / 5 width = 1.0
    }

    #[test]
    fn sumset_density_sparse_sumset() {
        // sumset = [1, 10, 100, 1000] in bins of width 100 over [1, 1000]:
        // [1,100], [101,200], ..., [901,1000] = 10 bins.
        let sumset = vec![1u64, 10, 100, 1000];
        let d = sumset_density(&sumset, 100);
        assert_eq!(d.len(), 10);
        // bin [1, 100] has {1, 10, 100} = 3 / 100
        assert_eq!(d[0].0, 1);
        assert!((d[0].1 - 3.0 / 100.0).abs() < 1e-12);
        // bin [901, 1000] has {1000} = 1 / 100
        assert_eq!(d[9].0, 901);
        assert!((d[9].1 - 1.0 / 100.0).abs() < 1e-12);
        // intermediate bins are empty
        for entry in &d[1..9] {
            assert_eq!(entry.1, 0.0);
        }
    }

    #[test]
    fn sumset_density_zero_width_or_empty() {
        assert!(sumset_density(&[], 10).is_empty());
        assert!(sumset_density(&[1, 2, 3], 0).is_empty());
    }

    #[test]
    fn sumset_density_singleton() {
        let d = sumset_density(&[42u64], 10);
        assert_eq!(d, vec![(42, 1.0)]);
    }

    #[test]
    fn notebook_pipeline_end_to_end_small_scale() {
        let denoms = standard_denoms_in_range(10, 1_000);
        assert!(!denoms.is_empty());
        let sumset = radix_sumset(&denoms, 3, 2_000);
        assert!(!sumset.is_empty());
        let gaps = radix_gaps(&sumset, 2);
        assert_eq!(
            approximate_from_below(&sumset, *sumset.last().unwrap()),
            sumset.last().copied()
        );
        let smallest = sumset[0];
        assert_eq!(approximate_from_below(&sumset, smallest - 1), None);
        assert!(gaps.len() < sumset.len());
    }

    /// At k=6 with default denom set, all gaps fall below dust@1sat/vb (~30s, ~500MB).
    #[test]
    #[ignore = "slow: full default config"]
    fn notebook_k6_gaps_above_dust_at_1_sat_vb_vanish() {
        let denoms = standard_denoms_in_range(DEFAULT_MIN_DENOM_SATS, DEFAULT_MAX_DENOM_SATS);
        let sumset = radix_sumset(&denoms, 6, DEFAULT_MAX_COMBINATION_VALUE_SATS);
        let dust_at_1_sat_vb = dust_at_feerate(P2WPKH_OUTPUT_VBYTES, 1.0);
        let max_gap = radix_gaps(&sumset, 1).into_iter().max().unwrap_or(0);
        assert!(
            max_gap < dust_at_1_sat_vb,
            "k=6 should drive all gaps below dust@1sat/vb={dust_at_1_sat_vb}; got {max_gap}"
        );
    }
}
