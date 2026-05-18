//! Structural classification of input value sets.

use crate::count::denoms::is_standard_denom;
use std::collections::HashMap;

/// Fraction of standard denominations; `0.0` on empty.
#[must_use]
pub fn radix_density(tx_values: &[u64]) -> f64 {
    if tx_values.is_empty() {
        return 0.0;
    }
    let hits = tx_values.iter().filter(|&&v| is_standard_denom(v)).count();
    hits as f64 / tx_values.len() as f64
}

/// Priority-ordered shape; each branch short-circuits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputShape {
    /// One value accounts for > 50% of the set.
    EqualAmount,
    /// > 50% of values are standard denominations.
    Radix,
    /// Sorted-diff CV below [`ARITHMETIC_CV`].
    Arithmetic,
    Mixed,
}

pub const ARITHMETIC_CV: f64 = 0.15;

impl InputShape {
    /// Classify `a` into a shape; `None` when `a.len() < 2`.
    #[must_use]
    pub fn of(a: &[u64]) -> Option<Self> {
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
            return Some(Self::EqualAmount);
        }

        let radix_count = a.iter().filter(|&&v| is_standard_denom(v)).count();
        if 2 * radix_count > n {
            return Some(Self::Radix);
        }

        let mut sorted = a.to_vec();
        sorted.sort_unstable();
        let diffs: Vec<f64> = sorted.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
        let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        if mean > 0.0 {
            let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64;
            let cv = var.sqrt() / mean;
            if cv < ARITHMETIC_CV {
                return Some(Self::Arithmetic);
            }
        }

        Some(Self::Mixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_density_fraction() {
        let tx = vec![1_000u64, 2_000, 5_000, 77, 13];
        let d = radix_density(&tx);
        assert!((d - 3.0 / 5.0).abs() < 1e-12);
    }

    #[test]
    fn radix_density_empty_is_zero() {
        assert_eq!(radix_density(&[]), 0.0);
    }

    #[test]
    fn equal_amount_dominates() {
        let a = vec![100_000u64; 10];
        assert_eq!(InputShape::of(&a), Some(InputShape::EqualAmount));
    }

    #[test]
    fn equal_amount_requires_strict_majority() {
        let a = vec![1u64, 1, 2, 3];
        assert_ne!(InputShape::of(&a), Some(InputShape::EqualAmount));
    }

    #[test]
    fn radix_detected() {
        let a = vec![1_000u64, 2_000, 5_000, 10_000, 20_000, 50_000, 77];
        assert_eq!(InputShape::of(&a), Some(InputShape::Radix));
    }

    #[test]
    fn arithmetic_progression_detected() {
        let a: Vec<u64> = (1..=20).map(|i| i * 1000).collect();
        assert_eq!(InputShape::of(&a), Some(InputShape::Arithmetic));
    }

    #[test]
    fn mixed_falls_through() {
        let a = vec![37u64, 891, 2, 15_333, 401, 7, 62_511, 88, 3_003, 19];
        assert_eq!(InputShape::of(&a), Some(InputShape::Mixed));
    }

    #[test]
    fn too_small_is_none() {
        assert!(InputShape::of(&[]).is_none());
        assert!(InputShape::of(&[42]).is_none());
    }
}
