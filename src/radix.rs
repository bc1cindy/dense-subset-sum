//! Radix-like denomination detection + ambiguity rewards.

use std::collections::HashSet;
use std::sync::LazyLock;

/// 20_999_999.9769 BTC · 10⁸ sat/BTC. Upper bound for the two denomination
/// series; values above are unreachable on-chain.
const MAX_MONEY_SATS: u64 = 2_099_999_997_690_000;

/// Preferred-value denominations everyone is likely to pick. Static lookup,
/// not a computed Hamming-weight function: HW has false positives (e.g.
/// 0xE0 = 128+64+32) and doesn't extend to the 1-2-5 decimal engineering
/// series. Membership is the whole criterion — there is no "degree" of
/// distinguishedness.
///
/// Seeds:
/// - `{1,2,5}·10^k` decimal engineering series;
/// - `{1,3}·2^k`    binary + 3-multiplier series.
///
/// Wasabi2 empirical denominations fit inside the binary series.
pub static DISTINGUISHED: LazyLock<HashSet<u64>> = LazyLock::new(|| {
    let mut s = HashSet::new();
    let mut p = 1u64;
    while p <= MAX_MONEY_SATS / 5 {
        for d in [1u64, 2, 5] {
            s.insert(d * p);
        }
        p *= 10;
    }
    let mut p = 1u64;
    loop {
        s.insert(p);
        if let Some(triple) = p.checked_mul(3) {
            s.insert(triple);
        }
        match p.checked_mul(2) {
            Some(next) if next <= MAX_MONEY_SATS => p = next,
            _ => break,
        }
    }
    s
});

pub fn is_distinguished(v: u64) -> bool {
    DISTINGUISHED.contains(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distinguished_contains_both_series() {
        for v in [1u64, 2, 5, 10, 20, 50, 100, 1_000, 100_000_000] {
            assert!(is_distinguished(v), "decimal 1-2-5 missing: {}", v);
        }
        for v in [1u64, 3, 4, 6, 8, 12, 16, 24, 48, 1024, 3072] {
            assert!(is_distinguished(v), "binary 1-3 missing: {}", v);
        }
    }

    #[test]
    fn test_distinguished_rejects_arbitrary_values() {
        // HW-based detection would false-positive on 0xE0 (3 bits in base-2); lookup rejects it.
        for v in [7u64, 0xE0, 12_345, 67_890, 999_999, 123_456_789] {
            assert!(!is_distinguished(v), "arbitrary value accepted: {}", v);
        }
    }

    #[test]
    fn test_distinguished_rejects_zero() {
        assert!(!is_distinguished(0));
    }

    #[test]
    fn test_distinguished_set_is_small() {
        // Point of the set: few values so multiplicity is high. Keep it bounded.
        assert!(DISTINGUISHED.len() < 200, "grew to {}", DISTINGUISHED.len());
    }
}
