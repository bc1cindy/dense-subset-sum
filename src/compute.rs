//! Counting paths over a CoinJoin transaction. All return [`Ambiguity`].

use crate::Ambiguity;
use crate::count::denoms::standard_denoms_in_range;
use crate::count::oracle::{BruteError, brute_force_w_restricted};
use crate::count::radix::{
    DEFAULT_MAX_DENOM_SATS, DEFAULT_MIN_DENOM_SATS, radix_decompose, radix_mapping_count,
};
use std::collections::{HashMap, HashSet};

/// Default cap on subset size M. Beyond 5, `C(N, M)` (binomial) grows fast enough that
/// the count is dominated by combinatorial inflation rather than information about distinct
/// payment interpretations; callers may override per problem.
pub const KNEE: usize = 5;

/// Non-empty output subset sums; `None` when `outputs.len() > 63` exceeds u64 mask width.
pub(crate) fn output_subsums(outputs: &[u64]) -> Option<HashSet<u64>> {
    let n = outputs.len();
    if n > 63 {
        return None;
    }
    let mut sums = HashSet::with_capacity(1usize << n);
    for mask in 1u64..(1u64 << n) {
        let s: u64 = (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| outputs[i])
            .sum();
        sums.insert(s);
    }
    Some(sums)
}

/// Exact `Σ_{E ∈ output_subsums} Σ_{m=1..=max_size} W(m, E)`; excludes the trivial
/// full-input mapping. `Ambiguity::Unknown` when `N > 20`. `max_size` caps subset size M.
#[must_use]
pub fn w_brute(inputs: &[u64], outputs: &[u64], max_size: usize) -> Ambiguity {
    if inputs.is_empty() || outputs.is_empty() || max_size == 0 {
        return Ambiguity::Exact(0);
    }
    let Some(targets) = output_subsums(outputs) else {
        return Ambiguity::Unknown;
    };
    let n_in = inputs.len();
    let full_input_sum: u64 = inputs.iter().sum();
    let cap = max_size.min(n_in);
    let mut count: u128 = 0;
    for &target in &targets {
        for m in 1..=cap {
            match brute_force_w_restricted(inputs, m, target) {
                Ok(w) => {
                    let mut delta = w;
                    if m == n_in && target == full_input_sum {
                        delta = delta.saturating_sub(1);
                    }
                    count = count.saturating_add(delta);
                }
                Err(BruteError::TooLarge) => return Ambiguity::Unknown,
                Err(BruteError::SumOverflow) => continue,
            }
        }
    }
    Ambiguity::Exact(count)
}

/// `Σ_outputs k × m!` where `k` = distinct denoms, `m` = min multiplicity in `outputs`; `max_size`
/// caps denoms per decomposition. Non-decomposable outputs contribute 0.
#[must_use]
pub fn radix_mappings(outputs: &[u64], max_size: usize) -> Ambiguity {
    if outputs.is_empty() || max_size == 0 {
        return Ambiguity::Exact(0);
    }
    let denoms = standard_denoms_in_range(DEFAULT_MIN_DENOM_SATS, DEFAULT_MAX_DENOM_SATS);
    let mut output_mult: HashMap<u64, usize> = HashMap::new();
    for &v in outputs {
        *output_mult.entry(v).or_insert(0) += 1;
    }
    let mut total: u128 = 0;
    for &output in outputs {
        let Some(decomp) = radix_decompose(&denoms, output, max_size) else {
            continue;
        };
        let k_distinct = decomp.iter().collect::<HashSet<_>>().len();
        let m_min = decomp
            .iter()
            .map(|d| output_mult.get(d).copied().unwrap_or(1))
            .min()
            .unwrap_or(1);
        if let Some(mappings) = radix_mapping_count(k_distinct, m_min) {
            total = total.saturating_add(mappings);
        }
    }
    Ambiguity::Exact(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn w_brute_empty_is_zero() {
        assert_eq!(w_brute(&[], &[5, 10], 8), Ambiguity::Exact(0));
        assert_eq!(w_brute(&[5, 10], &[], 8), Ambiguity::Exact(0));
        assert_eq!(w_brute(&[1, 2], &[3], 0), Ambiguity::Exact(0));
    }

    #[test]
    fn w_brute_excludes_trivial_full_input() {
        assert_eq!(w_brute(&[500], &[500], 8), Ambiguity::Exact(0));
    }

    #[test]
    fn w_brute_simple_match() {
        assert_eq!(w_brute(&[500, 300], &[500], 8), Ambiguity::Exact(1));
        assert_eq!(w_brute(&[500, 300], &[500, 300], 8), Ambiguity::Exact(2));
    }

    #[test]
    fn w_brute_n_above_20_is_unknown() {
        let inputs: Vec<u64> = (1..=21).collect();
        assert_eq!(w_brute(&inputs, &[10], 5), Ambiguity::Unknown);
    }

    #[test]
    fn radix_mappings_empty_or_zero_is_zero() {
        assert_eq!(radix_mappings(&[], 6), Ambiguity::Exact(0));
        assert_eq!(radix_mappings(&[1000], 0), Ambiguity::Exact(0));
    }

    #[test]
    fn radix_mappings_single_denom_output() {
        assert_eq!(radix_mappings(&[1000], 6), Ambiguity::Exact(1));
    }

    #[test]
    fn radix_mappings_repeated_denom_increases_m() {
        assert_eq!(radix_mappings(&[1000, 1000], 6), Ambiguity::Exact(4));
    }

    proptest! {
        #[test]
        fn w_brute_monotonic_in_max_size(
            inputs in prop::collection::vec(1u64..=100, 1..=6),
            outputs in prop::collection::vec(1u64..=100, 1..=4),
            d1 in 1usize..=6,
            d2 in 1usize..=6,
        ) {
            let (low, high) = (d1.min(d2), d1.max(d2));
            let c_low = w_brute(&inputs, &outputs, low).lower_bound_count().unwrap_or(0);
            let c_high = w_brute(&inputs, &outputs, high).lower_bound_count().unwrap_or(0);
            prop_assert!(c_high >= c_low);
        }

        #[test]
        fn w_brute_zero_max_size_is_exact_zero(
            inputs in prop::collection::vec(1u64..=100, 1..=6),
            outputs in prop::collection::vec(1u64..=100, 1..=4),
        ) {
            prop_assert_eq!(w_brute(&inputs, &outputs, 0), Ambiguity::Exact(0));
        }

        #[test]
        fn radix_mappings_monotonic_in_k(
            outputs in prop::collection::vec(1u64..=1_000_000, 1..=4),
            k1 in 1usize..=6,
            k2 in 1usize..=6,
        ) {
            let (low, high) = (k1.min(k2), k1.max(k2));
            let m_low = radix_mappings(&outputs, low).lower_bound_count().unwrap_or(0);
            let m_high = radix_mappings(&outputs, high).lower_bound_count().unwrap_or(0);
            prop_assert!(m_high >= m_low);
        }

        #[test]
        fn radix_mappings_always_exact(
            outputs in prop::collection::vec(1u64..=1_000_000, 1..=4),
            k in 1usize..=6,
        ) {
            prop_assert!(matches!(radix_mappings(&outputs, k), Ambiguity::Exact(_)));
        }

        #[test]
        fn radix_mappings_zero_max_size_is_exact_zero(
            outputs in prop::collection::vec(1u64..=100, 1..=4),
        ) {
            prop_assert_eq!(radix_mappings(&outputs, 0), Ambiguity::Exact(0));
        }
    }
}
