//! Preferred-value denominations: `{2^k} ∪ {1,2}·3^k ∪ {1,2,5}·10^k`.
//! Source: "Small Hamming Weight Denominations for CoinJoins".

use crate::count::radix::MAX_SATS;
use std::collections::{BTreeSet, HashSet};
use std::sync::LazyLock;

/// `{2^k} ∪ {1,2}·3^k ∪ {1,2,5}·10^k`. Static lookup avoids HW false-positives like `0xE0`.
pub static STANDARD_DENOMS: LazyLock<HashSet<u64>> = LazyLock::new(|| {
    let mut s = HashSet::new();
    let mut p = 1u64;
    while p <= MAX_SATS / 5 {
        for d in [1u64, 2, 5] {
            s.insert(d * p);
        }
        p = match p.checked_mul(10) {
            Some(n) => n,
            None => break,
        };
    }
    let mut p = 1u64;
    loop {
        s.insert(p);
        match p.checked_mul(2) {
            Some(next) if next <= MAX_SATS => p = next,
            _ => break,
        }
    }
    let mut p = 1u64;
    loop {
        s.insert(p);
        if let Some(double) = p.checked_mul(2) {
            if double <= MAX_SATS {
                s.insert(double);
            }
        }
        match p.checked_mul(3) {
            Some(next) if next <= MAX_SATS => p = next,
            _ => break,
        }
    }
    s
});

#[must_use]
pub fn is_standard_denom(v: u64) -> bool {
    STANDARD_DENOMS.contains(&v)
}

/// `binary ∪ ternary ∪ decimal` in `[min_denom, max_denom]`, ascending. Powers `b^k < min_denom`
/// are dropped before multiplication, so e.g. `2·3^5 = 486 ∉ result(294, 1e8)`.
///
/// ```
/// use dense_subset_sum::standard_denoms_in_range;
/// let d = standard_denoms_in_range(1, 20);
/// assert!(d.contains(&1) && d.contains(&8) && d.contains(&10));
/// assert!(!d.contains(&7));
/// ```
#[must_use]
pub fn standard_denoms_in_range(min_denom: u64, max_denom: u64) -> Vec<u64> {
    if max_denom == 0 || max_denom < min_denom {
        return Vec::new();
    }
    let mut s: BTreeSet<u64> = BTreeSet::new();
    s.extend(binary_denoms_in_range(min_denom, max_denom));
    s.extend(ternary_denoms_in_range(min_denom, max_denom));
    s.extend(decimal_denoms_in_range(min_denom, max_denom));
    s.into_iter().collect()
}

/// Powers of `b` in `[min, max]`, ascending. Empty when `b < 2`.
#[must_use]
pub fn powers_in_range(b: u64, min: u64, max: u64) -> Vec<u64> {
    if b < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut p = 1u64;
    while p < min {
        match p.checked_mul(b) {
            Some(n) => p = n,
            None => return out,
        }
    }
    while p <= max {
        out.push(p);
        match p.checked_mul(b) {
            Some(n) => p = n,
            None => break,
        }
    }
    out
}

/// Sorted, deduped products `c·v` filtered to `[min, max]`.
#[must_use]
pub fn multiples_in_range(values: &[u64], coefficients: &[u64], min: u64, max: u64) -> Vec<u64> {
    let mut s: BTreeSet<u64> = BTreeSet::new();
    for &v in values {
        for &c in coefficients {
            if let Some(cv) = v.checked_mul(c) {
                if cv >= min && cv <= max {
                    s.insert(cv);
                }
            }
        }
    }
    s.into_iter().collect()
}

/// `{2^k}` in `[min, max]`.
#[must_use]
pub fn binary_denoms_in_range(min: u64, max: u64) -> Vec<u64> {
    multiples_in_range(&powers_in_range(2, min, max), &[1], min, max)
}

/// `{1, 2}·3^k` in `[min, max]`.
#[must_use]
pub fn ternary_denoms_in_range(min: u64, max: u64) -> Vec<u64> {
    multiples_in_range(&powers_in_range(3, min, max), &[1, 2], min, max)
}

/// `{1, 2, 5}·10^k` in `[min, max]`.
#[must_use]
pub fn decimal_denoms_in_range(min: u64, max: u64) -> Vec<u64> {
    multiples_in_range(&powers_in_range(10, min, max), &[1, 2, 5], min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_denoms_covers_decimal_125() {
        for v in [1u64, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 100_000_000] {
            assert!(is_standard_denom(v), "decimal {{1,2,5}}·10^k missing: {v}");
        }
    }

    #[test]
    fn standard_denoms_covers_binary_powers() {
        for v in [1u64, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 1_048_576] {
            assert!(is_standard_denom(v), "binary 2^k missing: {v}");
        }
    }

    #[test]
    fn standard_denoms_covers_ternary_12() {
        for v in [1u64, 3, 9, 27, 81, 243, 729, 2_187] {
            assert!(is_standard_denom(v), "ternary 1·3^k missing: {v}");
        }
        for v in [2u64, 6, 18, 54, 162, 486, 1_458, 4_374] {
            assert!(is_standard_denom(v), "ternary 2·3^k missing: {v}");
        }
    }

    #[test]
    fn standard_denoms_rejects_three_times_pow2() {
        for v in [12u64, 24, 48, 96, 192, 384, 768, 1_536, 3_072, 6_144] {
            assert!(!is_standard_denom(v), "{v} = 3·2^k must not be standard");
        }
    }

    #[test]
    fn standard_denoms_rejects_arbitrary() {
        // HW-based detection would false-positive on 0xE0; static lookup rejects.
        for v in [7u64, 0xE0, 12_345, 999_999, 123_456_789] {
            assert!(!is_standard_denom(v), "arbitrary value accepted: {v}");
        }
        assert!(!is_standard_denom(0));
    }

    #[test]
    fn standard_denoms_in_range_default_params_matches_notebook() {
        let combined: HashSet<u64> = standard_denoms_in_range(294, 100_000_000)
            .into_iter()
            .collect();
        assert!(combined.contains(&512));
        assert!(combined.contains(&67_108_864));
        assert!(!combined.contains(&256), "2^8 < dust must be excluded");
        assert!(combined.contains(&729));
        assert!(combined.contains(&1_458));
        assert!(
            !combined.contains(&486),
            "486 = 2·3^5: 3^5 < dust drops the power before multiplying"
        );
        assert!(combined.contains(&1_000));
        assert!(combined.contains(&5_000));
        assert!(combined.contains(&100_000_000));
        assert!(!combined.contains(&500_000_000), "5·10^8 > max_denom");
        assert!(!combined.contains(&200_000_000), "2·10^8 > max_denom");
        for v in [768u64, 1_536, 3_072, 6_144, 12_288] {
            assert!(!combined.contains(&v), "{v} = 3·2^k must not appear");
        }
    }

    #[test]
    fn standard_denoms_in_range_empty_when_max_below_min() {
        assert!(standard_denoms_in_range(1_000, 500).is_empty());
        assert!(standard_denoms_in_range(0, 0).is_empty());
    }

    #[test]
    fn standard_denoms_in_range_is_sorted() {
        let v = standard_denoms_in_range(1, 10_000);
        for w in v.windows(2) {
            assert!(w[0] < w[1], "{} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn powers_in_range_matches_notebook_b2_dust_1e8() {
        // ceil(log_2(294)) = 9, floor(log_2(1e8)) = 26
        let p = powers_in_range(2, 294, 100_000_000);
        assert_eq!(p.first(), Some(&512));
        assert_eq!(p.last(), Some(&67_108_864));
        assert_eq!(p.len(), 18);
    }

    #[test]
    fn powers_in_range_matches_notebook_b3_dust_1e8() {
        // 3^5 = 243 < 294, so smallest is 3^6 = 729. 3^16 = 43_046_721 ≤ 1e8.
        let p = powers_in_range(3, 294, 100_000_000);
        assert_eq!(p.first(), Some(&729));
        assert_eq!(p.last(), Some(&43_046_721));
        assert_eq!(p.len(), 11);
    }

    #[test]
    fn powers_in_range_b_below_2_is_empty() {
        assert!(powers_in_range(0, 1, 100).is_empty());
        assert!(powers_in_range(1, 1, 100).is_empty());
    }

    #[test]
    fn multiples_in_range_matches_notebook_decimal_125() {
        let p = powers_in_range(10, 294, 100_000_000);
        let m = multiples_in_range(&p, &[1, 2, 5], 294, 100_000_000);
        assert!(m.contains(&1_000));
        assert!(m.contains(&5_000));
        assert!(m.contains(&100_000_000));
        assert!(m.contains(&50_000_000));
        assert!(!m.contains(&200_000_000));
        assert!(!m.contains(&500_000_000));
    }

    #[test]
    fn multiples_in_range_filters_below_min() {
        let m = multiples_in_range(&[100, 200], &[1, 2], 150, 1_000);
        // 1·100=100<150 ✗, 1·200=200 ✓, 2·100=200 ✓ (dedup), 2·200=400 ✓
        assert_eq!(m, vec![200, 400]);
    }

    #[test]
    fn multiples_in_range_is_sorted_deduplicated() {
        let m = multiples_in_range(&[2, 3, 6], &[1, 2, 3], 1, 100);
        // {2,3,6, 4,6,12, 6,9,18} → sorted unique {2,3,4,6,9,12,18}
        assert_eq!(m, vec![2, 3, 4, 6, 9, 12, 18]);
    }

    #[test]
    fn binary_denoms_in_range_default_params() {
        let b = binary_denoms_in_range(294, 100_000_000);
        assert_eq!(b.len(), 18);
        assert_eq!(b[0], 512);
        assert_eq!(b[17], 67_108_864);
    }

    #[test]
    fn ternary_denoms_in_range_default_params() {
        let t = ternary_denoms_in_range(294, 100_000_000);
        assert!(t.contains(&729));
        assert!(t.contains(&1458));
        assert!(t.contains(&43_046_721));
        assert!(t.contains(&86_093_442));
        assert!(!t.contains(&486));
    }

    #[test]
    fn decimal_denoms_in_range_default_params() {
        let d = decimal_denoms_in_range(294, 100_000_000);
        assert!(d.contains(&1_000));
        assert!(d.contains(&5_000));
        assert!(d.contains(&100_000_000));
        assert!(!d.contains(&200_000_000));
        assert!(!d.contains(&500_000_000));
    }

    #[test]
    fn standard_denoms_in_range_equals_union_of_three_bases() {
        let (lo, hi) = (294, 100_000_000);
        let combined: HashSet<u64> = standard_denoms_in_range(lo, hi).into_iter().collect();
        let b: HashSet<u64> = binary_denoms_in_range(lo, hi).into_iter().collect();
        let t: HashSet<u64> = ternary_denoms_in_range(lo, hi).into_iter().collect();
        let d: HashSet<u64> = decimal_denoms_in_range(lo, hi).into_iter().collect();
        let union: HashSet<u64> = b
            .union(&t)
            .copied()
            .collect::<HashSet<u64>>()
            .union(&d)
            .copied()
            .collect();
        assert_eq!(combined, union);
    }
}
