//! Structural classifier for input sets: labels a value vector by its
//! dominant algebraic shape (equal-amount, radix-geometric, arithmetic, or
//! none of the above). Used to calibrate the W-based penalty terms and to
//! pick the appropriate test fixture family.

use std::collections::HashMap;

use crate::radix::is_distinguished;

/// Empirical shape of an input set. Priority-ordered; each branch short-circuits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmpiricalRegime {
    /// A single value accounts for > 50% of the set.
    EqualAmount,
    /// > 50% of values are preferred denominations ({1,2,5}·10^k ∪ {1,3}·2^k).
    RadixGeometric,
    /// Sorted adjacent-diff coefficient of variation below `ARITHMETIC_CV`.
    Arithmetic,
    /// Fallback: wide/unstructured batch.
    PathologicalBatch,
}

/// CV threshold below which sorted adjacent differences look arithmetic.
pub const ARITHMETIC_CV: f64 = 0.15;

/// Classify `a` into one of the four regimes. `None` if `a` is too small
/// (fewer than 2 values) to meaningfully classify.
pub fn empirical_regime(a: &[u64]) -> Option<EmpiricalRegime> {
    if a.len() < 2 {
        return None;
    }
    let n = a.len();

    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &v in a {
        *counts.entry(v).or_insert(0) += 1;
    }
    let max_mult = counts.values().copied().max().unwrap_or(0);
    if 2 * max_mult > n {
        return Some(EmpiricalRegime::EqualAmount);
    }

    let radix_count = a.iter().filter(|&&v| is_distinguished(v)).count();
    if 2 * radix_count > n {
        return Some(EmpiricalRegime::RadixGeometric);
    }

    let mut sorted = a.to_vec();
    sorted.sort_unstable();
    let diffs: Vec<f64> = sorted.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    if mean > 0.0 {
        let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64;
        let cv = var.sqrt() / mean;
        if cv < ARITHMETIC_CV {
            return Some(EmpiricalRegime::Arithmetic);
        }
    }

    Some(EmpiricalRegime::PathologicalBatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_amount_dominates() {
        let a = vec![100_000u64; 10];
        assert_eq!(empirical_regime(&a), Some(EmpiricalRegime::EqualAmount));
    }

    #[test]
    fn equal_amount_requires_strict_majority() {
        let a = vec![1u64, 1, 2, 3]; // mult 2 of 4: not > half
        assert_ne!(empirical_regime(&a), Some(EmpiricalRegime::EqualAmount));
    }

    #[test]
    fn radix_geometric_detected() {
        let a = vec![1_000u64, 2_000, 5_000, 10_000, 20_000, 50_000, 77];
        assert_eq!(empirical_regime(&a), Some(EmpiricalRegime::RadixGeometric));
    }

    #[test]
    fn arithmetic_progression_detected() {
        let a: Vec<u64> = (1..=20).map(|i| i * 1000).collect();
        assert_eq!(empirical_regime(&a), Some(EmpiricalRegime::Arithmetic));
    }

    #[test]
    fn pathological_falls_through() {
        // Deliberately chaotic: mixed magnitudes, no dominant denomination.
        let a = vec![37u64, 891, 2, 15_333, 401, 7, 62_511, 88, 3_003, 19];
        assert_eq!(
            empirical_regime(&a),
            Some(EmpiricalRegime::PathologicalBatch)
        );
    }

    #[test]
    fn too_small_is_none() {
        assert!(empirical_regime(&[]).is_none());
        assert!(empirical_regime(&[42]).is_none());
    }
}
