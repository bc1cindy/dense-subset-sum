//! Four counting paths over a CoinJoin transaction. All return [`Ambiguity`]:
//! [`w_brute`], [`radix_mappings`], [`w_sparse`], [`w_sasamoto`].

use crate::Ambiguity;
use crate::count::companion::sasamoto_approx;
use crate::count::denoms::standard_denoms_in_range;
use crate::count::oracle::{BruteError, DpError, brute_force_w_restricted, dp_w_restricted};
use crate::count::radix::{
    DEFAULT_MAX_DENOM_SATS, DEFAULT_MIN_DENOM_SATS, radix_decompose, radix_mapping_count,
};
use crate::count::sparse_conv::Goldilocks;
use crate::count::sumset::{Bound, GradedSumset, GradedSumsetBudget};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

/// Default cap on subset size M. Beyond 5, `C(N, M)` (binomial) grows fast enough that
/// the count is dominated by combinatorial inflation rather than information about distinct
/// payment interpretations; callers may override per problem.
pub const KNEE: usize = 5;

/// Default `memory_budget` for [`w_sparse`]: 2^26 ≈ 67 M sumset entries (~1.5 GB at
/// ~24 bytes per entry), matching the upper bound suggested for sparse convolution before
/// switching strategies. Callers on memory-constrained targets should override.
pub const DEFAULT_MEMORY_BUDGET: NonZeroUsize = NonZeroUsize::new(1 << 26).unwrap();

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

/// Exact `Σ_{E ∈ output_subsums} Σ_{m=1..=max_size} W(m, E)` via the pseudo-polynomial DP oracle
/// [`dp_w_restricted`]; excludes the trivial full-input mapping, same as [`w_brute`]/[`w_sparse`].
/// Covers small-reachable-sum transactions exactly where `w_brute`'s `N ≤ 20` enumeration can't
/// reach. `Ambiguity::Unknown` when `output_subsums` returns `None`, or when any DP call exceeds
/// `max_cells` (budget/size overflow) — never a wrong count or a panic.
#[must_use]
pub fn w_dp(inputs: &[u64], outputs: &[u64], max_size: usize, max_cells: usize) -> Ambiguity {
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
            match dp_w_restricted(inputs, m, target, max_cells) {
                Ok(w) => {
                    let mut delta = w;
                    if m == n_in && target == full_input_sum {
                        delta = delta.saturating_sub(1);
                    }
                    count = count.saturating_add(delta);
                }
                Err(DpError::ExceedsBudget | DpError::SumOverflow | DpError::EmptyOrAllZero) => {
                    return Ambiguity::Unknown;
                }
            }
        }
    }
    Ambiguity::Exact(count)
}

/// Count via sparse convolution (Bringmann/Fischer/Nakos arXiv:2107.07625). Returns
/// `Ambiguity::Exact` if no truncation, else `Ambiguity::LowerBound`. `max_size` caps subset size M.
#[must_use]
pub fn w_sparse(
    inputs: &[u64],
    outputs: &[u64],
    max_size: usize,
    memory_budget: NonZeroUsize,
) -> Ambiguity {
    if inputs.is_empty() || outputs.is_empty() || max_size == 0 {
        return Ambiguity::Exact(0);
    }
    let Some(targets_set) = output_subsums(outputs) else {
        return Ambiguity::Unknown;
    };
    let mut targets: Vec<u64> = targets_set.into_iter().collect();
    targets.sort_unstable();
    let n_in = inputs.len();
    let cap = max_size.min(n_in);
    let full_input_sum: u64 = inputs.iter().sum();
    let budget = GradedSumsetBudget::<Goldilocks>::default().with_max_size(memory_budget);
    let sumset: GradedSumset =
        GradedSumset::<Goldilocks>::builder(inputs, budget, &targets).bounded(cap);
    let mut count: u128 = 0;
    let mut bound = Bound::Exact;
    for &target in &targets {
        let c = sumset.count_total(target);
        count = count.saturating_add(u128::from(c.visible()));
        bound = bound.join(c.bound());
    }
    // Match w_brute's exclusion of the trivial full-input mapping.
    if cap == n_in && targets.binary_search(&full_input_sum).is_ok() {
        let trivial = sumset.count_at(n_in, full_input_sum);
        count = count.saturating_sub(u128::from(trivial.visible()));
    }
    (count, bound).into()
}

/// Saddle-point `log W(E)` peak over Dense-regime output subsums (Sasamoto cond-mat/0106125).
/// Approximate, never trustworthy as sole signal: cross-validate against [`w_brute`]/[`w_sparse`].
#[must_use]
pub fn w_sasamoto(inputs: &[u64], outputs: &[u64]) -> Ambiguity {
    if inputs.is_empty() || outputs.is_empty() {
        return Ambiguity::Unknown;
    }
    let Some(targets) = output_subsums(outputs) else {
        return Ambiguity::Unknown;
    };
    let mut peak: Option<f64> = None;
    let sum_a: u64 = inputs.iter().sum();
    for target in targets {
        if target == 0 || target >= sum_a {
            continue;
        }
        if let Some(log_w) = sasamoto_approx(inputs, target) {
            if log_w.is_finite() {
                peak = Some(peak.map_or(log_w, |p| p.max(log_w)));
            }
        }
    }
    peak.into()
}

/// Distinct non-empty output subset sums.
///
/// Computed by a reachability DP in O(n · |reachable|): start from {0} and, for each output, fold
/// in every existing sum plus that output. Feasible for dense/denominated outputs — subsets collide,
/// so the reachable set stays small even when n is large — where the old 2^n mask enumeration was
/// not (it OOM'd and capped at n > 63). Returns `None` when the reachable set exceeds the budget:
/// that is the sparse regime, where the subset-sum count is intractable anyway and callers degrade
/// to `Unknown`.
pub(crate) fn output_subsums(outputs: &[u64]) -> Option<HashSet<u64>> {
    const REACHABLE_BUDGET: usize = 1 << 22; // ~4M distinct sums ≈ tens of MB
    let mut sums: HashSet<u64> = HashSet::new();
    sums.insert(0);
    for &v in outputs {
        if v == 0 {
            continue; // adding 0 never reaches a new sum
        }
        let additions: Vec<u64> = sums.iter().map(|&s| s.saturating_add(v)).collect();
        sums.extend(additions);
        if sums.len() > REACHABLE_BUDGET {
            return None;
        }
    }
    sums.remove(&0); // exclude the empty subset
    Some(sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

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
    fn w_dp_matches_brute_on_small() {
        let cases: &[(&[u64], &[u64])] = &[
            (&[500, 300], &[500, 300]),
            (&[100, 200, 300], &[150, 150, 200, 100]),
            (&[500, 300], &[500]),
            (&[1, 2, 3, 4], &[1, 2, 3, 4]),
            (&[10, 20, 30], &[10, 20, 30, 40]),
        ];
        for &(inputs, outputs) in cases {
            let brute = w_brute(inputs, outputs, 8);
            let dp = w_dp(inputs, outputs, 8, 1_000_000);
            assert_eq!(dp, brute, "inputs={inputs:?} outputs={outputs:?}");
        }
        assert_eq!(w_brute(&[500, 300], &[500, 300], 8), Ambiguity::Exact(2));
        assert_eq!(
            w_dp(&[500, 300], &[500, 300], 8, 1_000_000),
            Ambiguity::Exact(2)
        );
    }

    #[test]
    fn w_dp_matches_sparse_on_small() {
        let cases: &[(&[u64], &[u64])] = &[
            (&[500, 300], &[500, 300]),
            (&[100, 200, 300], &[150, 150, 200, 100]),
            (&[500, 300], &[500]),
            (&[1, 2, 3, 4], &[1, 2, 3, 4]),
            (&[10, 20, 30], &[10, 20, 30, 40]),
        ];
        for &(inputs, outputs) in cases {
            let sparse = w_sparse(inputs, outputs, 8, DEFAULT_MEMORY_BUDGET);
            let dp = w_dp(inputs, outputs, 8, 1_000_000);
            assert_eq!(dp, sparse, "inputs={inputs:?} outputs={outputs:?}");
        }
    }

    #[test]
    fn w_dp_budget_exceeded_is_unknown() {
        let inputs: Vec<u64> = (1..=12).collect();
        let outputs: Vec<u64> = (1..=12).collect();
        assert_eq!(w_dp(&inputs, &outputs, 8, 4), Ambiguity::Unknown);
    }

    #[test]
    fn w_dp_empty_is_zero() {
        assert_eq!(w_dp(&[], &[5, 10], 8, 1_000_000), Ambiguity::Exact(0));
        assert_eq!(w_dp(&[5, 10], &[], 8, 1_000_000), Ambiguity::Exact(0));
        assert_eq!(w_dp(&[1], &[1], 0, 1_000_000), Ambiguity::Exact(0));
    }

    #[test]
    fn w_sparse_empty_is_zero_exact() {
        assert_eq!(w_sparse(&[], &[5, 10], 4, nz(1000)), Ambiguity::Exact(0));
        assert_eq!(w_sparse(&[5, 10], &[], 4, nz(1000)), Ambiguity::Exact(0));
        assert_eq!(w_sparse(&[1], &[1], 0, nz(1000)), Ambiguity::Exact(0));
    }

    #[test]
    fn w_sasamoto_empty_is_unknown() {
        assert_eq!(w_sasamoto(&[], &[5]), Ambiguity::Unknown);
        assert_eq!(w_sasamoto(&[5], &[]), Ambiguity::Unknown);
    }

    #[test]
    fn w_sasamoto_near_half_energy_is_unknown_not_panic() {
        // Regression: energy near ΣA/2 pushes the saddle-point α outside the finite solver bracket.
        // This used to panic in find_alpha via kappa_c_at; w_sasamoto must degrade to Unknown.
        let base: u64 = 21_000_000 * 100_000_000 / 400;
        let inputs: Vec<u64> = (0..100u64).map(|i| base + i).collect();
        let sum_a: u128 = inputs.iter().map(|&x| u128::from(x)).sum();
        let outputs = vec![(sum_a / 2) as u64];
        assert_eq!(w_sasamoto(&inputs, &outputs), Ambiguity::Unknown);
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

    fn naive_output_subsums(outputs: &[u64]) -> HashSet<u64> {
        let n = outputs.len();
        let mut sums = HashSet::new();
        for mask in 1u64..(1u64 << n) {
            let s: u64 = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| outputs[i])
                .sum();
            sums.insert(s);
        }
        sums
    }

    #[test]
    fn output_subsums_matches_naive_on_small_sets() {
        for outputs in [
            vec![1u64, 2, 3],
            vec![5, 5, 5, 5],
            vec![512, 512, 1024, 2048],
            vec![1, 10, 100, 1000, 10],
        ] {
            assert_eq!(
                output_subsums(&outputs).unwrap(),
                naive_output_subsums(&outputs)
            );
        }
    }

    #[test]
    fn output_subsums_scales_to_large_dense_set() {
        // 40 denominated outputs — far past the old 2^n / n<=63 enumeration — but few DISTINCT
        // subset sums because the values collide. The DP returns the exact set cheaply.
        let outputs: Vec<u64> = std::iter::repeat(131_072u64)
            .take(20)
            .chain(std::iter::repeat(262_144u64).take(20))
            .collect();
        let sums = output_subsums(&outputs).expect("dense set must be tractable");
        assert!(
            sums.len() < 2000,
            "expected few distinct sums, got {}",
            sums.len()
        );
        assert!(sums.contains(&131_072));
        assert!(sums.contains(&(20 * 131_072 + 20 * 262_144)));
    }

    #[test]
    fn output_subsums_returns_none_when_intractable() {
        // 40 distinct powers of two never collide, so the reachable set explodes past the budget —
        // the sparse regime. Returns None (the old enumeration would have hung at 2^40).
        let outputs: Vec<u64> = (1..=40u64).map(|i| 1u64 << i).collect();
        assert!(output_subsums(&outputs).is_none());
    }

    proptest! {
        #[test]
        fn output_subsums_dp_equals_naive(
            outputs in prop::collection::vec(1u64..=64, 1..=10),
        ) {
            prop_assert_eq!(output_subsums(&outputs).unwrap(), naive_output_subsums(&outputs));
        }

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
        fn w_dp_agrees_with_brute(
            inputs in prop::collection::vec(1u64..=30, 1..=8),
            outputs in prop::collection::vec(1u64..=30, 1..=8),
        ) {
            let brute = w_brute(&inputs, &outputs, inputs.len());
            let dp = w_dp(&inputs, &outputs, inputs.len(), 1_000_000);
            prop_assert_eq!(dp, brute);
        }

        #[test]
        fn w_sparse_le_w_brute(
            inputs in prop::collection::vec(1u64..=30, 2..=6),
            outputs in prop::collection::vec(1u64..=30, 1..=4),
        ) {
            let brute = w_brute(&inputs, &outputs, inputs.len()).lower_bound_count().unwrap_or(0);
            let sparse = w_sparse(&inputs, &outputs, inputs.len(), nz(1_000_000))
                .lower_bound_count().unwrap_or(0);
            prop_assert!(sparse <= brute);
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

        /// Empty/degenerate inputs always yield `Ambiguity::Unknown` from sasamoto.
        #[test]
        fn w_sasamoto_unknown_on_empty(
            outputs in prop::collection::vec(1u64..=1_000_000, 1..=8),
        ) {
            prop_assert_eq!(w_sasamoto(&[], &outputs), Ambiguity::Unknown);
            prop_assert_eq!(w_sasamoto(&outputs, &[]), Ambiguity::Unknown);
        }

        /// At tiny N (<=8), regime is Sparse/Transitional so sasamoto returns Unknown.
        #[test]
        fn w_sasamoto_unknown_at_tiny_n(
            inputs in prop::collection::vec(1u64..=1000, 2..=8),
            outputs in prop::collection::vec(1u64..=1000, 1..=4),
        ) {
            prop_assert_eq!(w_sasamoto(&inputs, &outputs), Ambiguity::Unknown);
        }
    }
}
