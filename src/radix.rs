//! Radix-like denomination detection + ambiguity rewards.

pub const RADIX_BASES: [u64; 3] = [2, 3, 10];

pub fn nonzero_digits_in_base(v: u64, base: u64) -> u32 {
    if base < 2 {
        return 0;
    }
    if v == 0 {
        return 0;
    }
    let mut count = 0u32;
    let mut v = v;
    while v > 0 {
        if !v.is_multiple_of(base) {
            count += 1;
        }
        v /= base;
    }
    count
}

pub fn is_radix_like_in_base(a: &[u64], base: u64, hw_threshold: u32) -> bool {
    if a.is_empty() {
        return false;
    }
    let low_hw_count = a
        .iter()
        .filter(|&&v| nonzero_digits_in_base(v, base) <= hw_threshold)
        .count();
    low_hw_count * 2 > a.len()
}

pub fn is_radix_like_any_base(a: &[u64], hw_threshold: u32) -> bool {
    RADIX_BASES
        .iter()
        .any(|&base| is_radix_like_in_base(a, base, hw_threshold))
}

pub fn distinguish_coins(a: &[u64], base: u64, hw_threshold: u32) -> (Vec<u64>, Vec<u64>) {
    let mut dist = Vec::new();
    let mut arb = Vec::new();
    for &v in a {
        if v > 0 && nonzero_digits_in_base(v, base) <= hw_threshold {
            dist.push(v);
        } else {
            arb.push(v);
        }
    }
    (dist, arb)
}

pub fn denomination_multiplicities(distinguished: &[u64]) -> Vec<(u64, usize)> {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for &v in distinguished {
        *counts.entry(v).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

/// k^⌈log_b(x)⌉ ambiguity; `k` = scarcest denom ≤ x (conservative).
pub fn coverage_bonus_log2(x: u64, base: u64, mults: &[(u64, usize)]) -> f64 {
    if x == 0 || base < 2 || mults.is_empty() {
        return 0.0;
    }
    let usable: Vec<usize> = mults
        .iter()
        .filter(|(d, _)| *d <= x)
        .map(|(_, m)| *m)
        .collect();
    if usable.is_empty() {
        return 0.0;
    }
    let k = *usable
        .iter()
        .min()
        .expect("usable non-empty (checked above)");
    if k <= 1 {
        return 0.0;
    }
    let digits = ((x as f64).ln() / (base as f64).ln()).ceil().max(1.0);
    digits * (k as f64).log2()
}

/// `log₂(k)` head (not `k-1`) matches log₂(C(k,⌊k/2⌋)) at k=2,3 — tightest sound bound.
pub fn denomination_reward_log2(mults: &[(u64, usize)]) -> f64 {
    let mut total = 0.0_f64;
    for (_, m) in mults {
        let m = *m as f64;
        let log_head = m.min(3.0);
        let extra = (m - 3.0).max(0.0);
        if log_head >= 1.0 {
            total += log_head.log2();
        }
        if extra > 0.0 {
            total += 0.5 * (extra + 1.0).log2();
        }
    }
    total
}

/// Weakly rewards size/distinctness of arbitrary values; 0.5·log₂(d+1).
pub fn arbitrary_distinctness_log2(arb: &[u64]) -> f64 {
    use std::collections::BTreeSet;
    let distinct: BTreeSet<u64> = arb.iter().filter(|&&v| v > 0).copied().collect();
    if distinct.is_empty() {
        0.0
    } else {
        0.5 * ((distinct.len() as f64) + 1.0).log2()
    }
}

pub fn best_radix_base(a: &[u64], hw_threshold: u32) -> Option<(u64, Vec<(u64, usize)>)> {
    let mut best: Option<(u64, Vec<(u64, usize)>)> = None;
    let mut best_count = 0usize;
    for &b in &RADIX_BASES {
        let (dist, _) = distinguish_coins(a, b, hw_threshold);
        if dist.len() > best_count {
            best_count = dist.len();
            best = Some((b, denomination_multiplicities(&dist)));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonzero_digits_in_base() {
        assert_eq!(nonzero_digits_in_base(0, 2), 0);
        assert_eq!(nonzero_digits_in_base(1, 2), 1);
        assert_eq!(nonzero_digits_in_base(7, 2), 3);
        assert_eq!(nonzero_digits_in_base(1024, 2), 1);
        assert_eq!(nonzero_digits_in_base(1024, 2), 1024u64.count_ones());

        assert_eq!(nonzero_digits_in_base(9, 3), 1);
        assert_eq!(nonzero_digits_in_base(10, 3), 2);
        assert_eq!(nonzero_digits_in_base(14_348_907, 3), 1);

        assert_eq!(nonzero_digits_in_base(10_000, 10), 1);
        assert_eq!(nonzero_digits_in_base(12_345, 10), 5);
        assert_eq!(nonzero_digits_in_base(100_000_000, 10), 1);

        assert_eq!(nonzero_digits_in_base(10, 1), 0);
        assert_eq!(nonzero_digits_in_base(10, 0), 0);
    }

    #[test]
    fn test_is_radix_like_any_base_detects_cross_base() {
        let powers_of_3 = vec![1u64, 3, 9, 27, 81, 243, 729, 2187];
        assert!(!is_radix_like_in_base(&powers_of_3, 2, 2));
        assert!(is_radix_like_in_base(&powers_of_3, 3, 1));
        assert!(is_radix_like_any_base(&powers_of_3, 1));

        let round_decimals = vec![10_000u64, 100_000, 1_000_000, 50_000, 200_000];
        assert!(is_radix_like_in_base(&round_decimals, 10, 1));
        assert!(is_radix_like_any_base(&round_decimals, 1));

        let organic = vec![12_345u64, 67_890, 23_456, 78_901, 34_567];
        assert!(!is_radix_like_any_base(&organic, 1));
    }

    #[test]
    fn test_radix_composite_manual_all_equal_denom() {
        let a: Vec<u64> = vec![8, 8, 8, 8];
        let (_dist, arb) = distinguish_coins(&a, 2, 1);
        assert_eq!(arb.len(), 0, "all 8s are base-2 hamming-1");
        let mults = denomination_multiplicities(&[8, 8, 8, 8]);
        assert_eq!(mults, vec![(8, 4)]);
        let reward = denomination_reward_log2(&mults);
        let expected = 3f64.log2() + 0.5;
        assert!(
            (reward - expected).abs() < 1e-10,
            "{} vs {}",
            reward,
            expected
        );
        let cov_below = coverage_bonus_log2(4, 2, &mults);
        assert_eq!(cov_below, 0.0);
        let cov_above = coverage_bonus_log2(12, 2, &mults);
        assert!((cov_above - 8.0).abs() < 1e-10, "got {}", cov_above);
    }

    #[test]
    fn test_radix_composite_picks_best_base() {
        let a: Vec<u64> = vec![100, 100, 1000, 1000, 10000];
        let (b, mults) = best_radix_base(&a, 1).expect("some base should win");
        assert_eq!(b, 10, "base 10 should win for decimal denominations");
        let total: usize = mults.iter().map(|(_, m)| m).sum();
        assert_eq!(total, 5);
    }

    /// Payoff is log2(k) linear for k≤3, sqrt-tail for k>3. Knee at k=3.
    #[test]
    fn test_denomination_reward_knee_at_k3() {
        let reward = |k: usize| denomination_reward_log2(&[(1, k)]);

        assert!((reward(1) - 0.0).abs() < 1e-12);
        assert!((reward(2) - 1.0).abs() < 1e-12);
        assert!((reward(3) - 3f64.log2()).abs() < 1e-12);

        for k in [4usize, 5, 7, 10, 20] {
            let expected = 3f64.log2() + 0.5 * ((k - 2) as f64).log2();
            assert!(
                (reward(k) - expected).abs() < 1e-12,
                "reward({}) = {} vs expected {}",
                k,
                reward(k),
                expected
            );
        }

        let marginals: Vec<f64> = (2..=10).map(|k| reward(k) - reward(k - 1)).collect();
        for w in marginals.windows(2) {
            assert!(
                w[0] > w[1],
                "marginal gains must strictly decrease (got {} → {})",
                w[0],
                w[1]
            );
        }

        let head_slope = (reward(3) - reward(1)) / 2.0;
        let tail_slope = (reward(10) - reward(4)) / 6.0;
        assert!(
            tail_slope < 0.3 * head_slope,
            "tail slope (k∈[4,10]) must be well below head slope (k∈[1,3]): {} vs {}",
            tail_slope,
            head_slope
        );
    }

    #[test]
    fn test_radix_composite_sublinear_past_k3() {
        let denoms: [u64; 4] = [1, 2, 4, 8];
        let target: u64 = 6;

        let composite = |k: usize| -> f64 {
            let a: Vec<u64> = denoms
                .iter()
                .flat_map(|&d| std::iter::repeat_n(d, k))
                .collect();
            let mults = denomination_multiplicities(&a);
            coverage_bonus_log2(target, 2, &mults) + denomination_reward_log2(&mults)
        };

        let marginal = |k: usize| composite(k + 1) - composite(k);

        let m12 = marginal(1);
        let m23 = marginal(2);
        let m34 = marginal(3);
        let m45 = marginal(4);
        assert!(m12 > m23, "k=1→2 ({}) must exceed k=2→3 ({})", m12, m23);
        assert!(m23 > m34, "k=2→3 ({}) must exceed k=3→4 ({})", m23, m34);
        assert!(m34 > m45, "k=3→4 ({}) must exceed k=4→5 ({})", m34, m45);

        assert!(
            m34 / m12 < 0.6,
            "composite knee must drop below 60% of the early step: m34/m12 = {}",
            m34 / m12
        );
    }

    #[test]
    fn test_arbitrary_distinctness_trivial_inputs_are_zero() {
        assert_eq!(arbitrary_distinctness_log2(&[]), 0.0);
        assert_eq!(arbitrary_distinctness_log2(&[0, 0, 0]), 0.0);
    }

    #[test]
    fn test_arbitrary_distinctness_closed_form() {
        assert!((arbitrary_distinctness_log2(&[12345]) - 0.5).abs() < 1e-12);
        assert!((arbitrary_distinctness_log2(&[7, 7, 7]) - 0.5).abs() < 1e-12);
        assert!((arbitrary_distinctness_log2(&[10, 20, 30]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_arbitrary_distinctness_filters_zeros() {
        let got = arbitrary_distinctness_log2(&[0, 10, 20]);
        let expected = 0.5 * 3f64.log2();
        assert!((got - expected).abs() < 1e-12, "got {}", got);
    }

    #[test]
    fn test_arbitrary_distinctness_sublinear_growth() {
        let b1 = arbitrary_distinctness_log2(&[1]);
        let b4 = arbitrary_distinctness_log2(&[1, 2, 3, 4]);
        let b16: Vec<u64> = (1..=16).collect();
        let b16 = arbitrary_distinctness_log2(&b16);
        assert!(b1 < b4 && b4 < b16, "monotone: {} {} {}", b1, b4, b16);
        assert!(b16 < 4.0 * b4, "sublinear: b4={} b16={}", b4, b16);
    }
}
