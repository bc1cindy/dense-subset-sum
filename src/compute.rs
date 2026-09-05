//! Five counting paths over a CoinJoin transaction, all returning [`Ambiguity`]:
//! [`w_brute`], [`w_dp`], [`w_sparse`], [`w_sasamoto`] compute `W(E) = #{S ⊆ A : ΣS = E}`
//! at increasing guarantee cost (exact enumeration → exact pseudo-poly DP → exact-or-truncated
//! sparse convolution → saddle-point approximation); [`radix_mappings`] computes the
//! denomination-mapping count `Σ k × m!` instead, and is tagged `Diagnostic` because that is a
//! different object. [`w_count`] is the dispatcher over the four `W(E)` paths: it tries them in
//! feasibility order, each self-gating to `Ambiguity::Unknown` when it can't handle the input,
//! and returns the first accepted result tagged with the [`Method`] that produced it. Acceptance
//! is not guarantee order — see [`w_count`].
//!
//! `log_w` is not one quantity across those paths. The exact tiers sum `W(E)` over *every*
//! distinct output subset sum and take the log of that total; [`w_sasamoto`] takes the largest
//! per-target `ln W(E)` instead, because a saddle-point estimate is a magnitude and summing
//! magnitudes across targets compounds the approximation. A sum-over-targets and a
//! peak-over-targets are comparable in order of magnitude, not entrywise.

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
    let Some(full_input_sum) = checked_total(inputs) else {
        return Ambiguity::Unknown;
    };
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

/// The inputs' total, or `None` when it does not fit a `u64`.
///
/// Saturating here would clamp to `u64::MAX` and hide exactly the condition the caller has to
/// detect: an input set this crate cannot count over. Every counting path already answers
/// "cannot handle this input" with [`Ambiguity::Unknown`], so the overflow answers the same way
/// rather than inventing a total no transaction has.
fn checked_total(inputs: &[u64]) -> Option<u64> {
    inputs.iter().copied().try_fold(0u64, u64::checked_add)
}

/// `Σ_{distinct output value} k × m!` where `k` = distinct denoms in the value's decomposition and
/// `m` = the smallest multiplicity those denoms have *in `outputs`*; `max_size` caps denoms per
/// decomposition.
///
/// The summand is a property of a value, not of each coin carrying it: a denomination repeated
/// `m` times contributes its `m!` permutations once, not `m` times. A decomposition naming a
/// denomination that is not itself an output contributes nothing — the exchange it stands for
/// swaps a size-`k` subset of outputs for a size-1 one, and a subset whose members are absent
/// cannot be swapped. Non-decomposable outputs contribute 0.
///
/// Returns [`Ambiguity::Diagnostic`]: this counts denomination exchanges among the outputs, which
/// is neither `W(E)` nor a bound on the transaction's mapping count — the call never sees the
/// inputs, and its answer moves under a uniform rescaling that leaves the mapping count fixed. It
/// is also incomplete by construction: permutations confined to one sub-transaction change no
/// assignment and should be divided back out, which needs the block structure this signature does
/// not carry.
#[must_use]
pub fn radix_mappings(outputs: &[u64], max_size: usize) -> Ambiguity {
    if outputs.is_empty() || max_size == 0 {
        return Ambiguity::Diagnostic(0);
    }
    let denoms = standard_denoms_in_range(DEFAULT_MIN_DENOM_SATS, DEFAULT_MAX_DENOM_SATS);
    let mut output_mult: HashMap<u64, usize> = HashMap::new();
    for &v in outputs {
        *output_mult.entry(v).or_insert(0) += 1;
    }
    let mut distinct: Vec<u64> = output_mult.keys().copied().collect();
    distinct.sort_unstable();
    let mut total: u128 = 0;
    for output in distinct {
        let Some(decomp) = radix_decompose(&denoms, output, max_size) else {
            continue;
        };
        let Some(m_min) = decomp
            .iter()
            .map(|d| output_mult.get(d).copied().unwrap_or(0))
            .min()
        else {
            continue; // empty decomposition: nothing to exchange
        };
        if m_min == 0 {
            continue;
        }
        // `radix_decompose` returns an ascending multiset.
        let k_distinct = 1 + decomp.windows(2).filter(|w| w[0] != w[1]).count();
        if let Some(mappings) = radix_mapping_count(k_distinct, m_min) {
            total = total.saturating_add(mappings);
        }
    }
    Ambiguity::Diagnostic(total)
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
    let Some(full_input_sum) = checked_total(inputs) else {
        return Ambiguity::Unknown;
    };
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
    let Some(full_input_sum) = checked_total(inputs) else {
        return Ambiguity::Unknown;
    };
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
    let Some(sum_a) = checked_total(inputs) else {
        return Ambiguity::Unknown;
    };
    for target in targets {
        if target == 0 || target >= sum_a {
            continue;
        }
        if let Some(log_w) = sasamoto_approx(inputs, target)
            && log_w.is_finite()
        {
            peak = Some(peak.map_or(log_w, |p| p.max(log_w)));
        }
    }
    peak.into()
}

/// Cap on `N` for [`w_brute`] within the [`w_count`] cascade: 2²⁰ subsets is instant.
pub const BRUTE_MAX: usize = 20;

/// Default `max_cells` budget for the [`w_dp`] tier within [`w_count`]: 2^26 ≈ 67M cells.
/// Each cell is a `u128` (16 bytes), so this caps the DP table at ~1 GB — the same order of
/// magnitude as [`DEFAULT_MEMORY_BUDGET`]'s ~1.5 GB sparse-conv budget, so neither tier is
/// the odd one out on resource use. Callers with a tighter/looser budget should call
/// `w_dp` directly with their own `max_cells`.
pub const DP_MAX_CELLS: usize = 1 << 26;

/// Which [`w_count`] tier produced a [`CountReport`]. `None` means every tier returned
/// `Ambiguity::Unknown`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    Brute,
    Dp,
    Sparse,
    Sasamoto,
    None,
}

/// Result of [`w_count`]: the ambiguity count/bound/approximation plus which tier produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CountReport {
    pub ambiguity: Ambiguity,
    pub method: Method,
}

/// Feasibility cascade over the four `W(E)` counting paths: [`w_brute`] → [`w_dp`] →
/// [`w_sparse`] → [`w_sasamoto`]. Each tier self-gates to `Ambiguity::Unknown` when it can't
/// handle the input (budget/regime).
///
/// The acceptance order is `Exact` → Dense `LogApprox` → `LowerBound` → `Unknown`, which is NOT
/// decreasing guarantee strength: a Dense saddle-point approximation is preferred over a
/// truncated `LowerBound` from the same input. That is deliberate and the reason is magnitude,
/// not guarantee — a lower bound that saturated far below the true count is a worse description
/// of a large dense mix than an approximation of its logarithm. Callers that need a strict floor
/// must read the variant, not the tier.
///
/// Counts all subset sizes (`max_size = inputs.len()`, i.e. the full `W(E)`).
#[must_use]
pub fn w_count(inputs: &[u64], outputs: &[u64]) -> CountReport {
    let max_size = inputs.len();

    if inputs.len() <= BRUTE_MAX {
        let a = w_brute(inputs, outputs, max_size);
        if !a.is_unknown() {
            return CountReport {
                ambiguity: a,
                method: Method::Brute,
            };
        }
    }

    let a = w_dp(inputs, outputs, max_size, DP_MAX_CELLS);
    if !a.is_unknown() {
        return CountReport {
            ambiguity: a,
            method: Method::Dp,
        };
    }

    // The last two tiers are one decision, not two tests: a truncated `LowerBound` describes a
    // large dense mix badly (the count saturates far below the truth), so an in-regime
    // saddle-point estimate is preferred over it — while an exact sparse count beats both. Matching
    // the pair says that in the shape of the values instead of recovering it from three bits.
    let sparse = w_sparse(inputs, outputs, max_size, DEFAULT_MEMORY_BUDGET);
    let sasamoto = w_sasamoto(inputs, outputs);
    let (ambiguity, method) = match (sparse, sasamoto) {
        (exact @ Ambiguity::Exact(_), _) => (exact, Method::Sparse),
        (_, approx @ Ambiguity::LogApprox(_)) => (approx, Method::Sasamoto),
        (floor @ Ambiguity::LowerBound(_), _) => (floor, Method::Sparse),
        _ => (Ambiguity::Unknown, Method::None),
    };
    CountReport { ambiguity, method }
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
    fn an_input_total_that_does_not_fit_is_refused_not_clamped() {
        // Saturating would report u64::MAX and let the count proceed over a total no transaction
        // has; the crate's answer to an input it cannot handle is Unknown.
        let overflowing = [u64::MAX, 1];
        assert_eq!(checked_total(&overflowing), None);
        assert_eq!(w_brute(&overflowing, &[1], 2), Ambiguity::Unknown);
        assert_eq!(
            w_dp(&overflowing, &[1], 2, DP_MAX_CELLS),
            Ambiguity::Unknown
        );
        assert_eq!(w_sasamoto(&overflowing, &[1]), Ambiguity::Unknown);
        assert_eq!(checked_total(&[1, 2, 3]), Some(6));
        assert_eq!(checked_total(&[]), Some(0));
    }

    #[test]
    fn radix_mappings_empty_or_zero_is_zero() {
        assert_eq!(radix_mappings(&[], 6), Ambiguity::Diagnostic(0));
        assert_eq!(radix_mappings(&[1000], 0), Ambiguity::Diagnostic(0));
    }

    #[test]
    fn radix_mappings_single_denom_output() {
        assert_eq!(radix_mappings(&[1000], 6), Ambiguity::Diagnostic(1));
    }

    #[test]
    fn radix_mappings_repeated_denom_counts_the_value_once() {
        // A value repeated m times is one value with m! permutations, not m values with m! each.
        assert_eq!(radix_mappings(&[1000, 1000], 6), Ambiguity::Diagnostic(2));
        assert_eq!(
            radix_mappings(&[5000, 5000, 5000], 6),
            Ambiguity::Diagnostic(6)
        );
    }

    #[test]
    fn radix_mappings_skips_decompositions_with_an_absent_denomination() {
        // 5512 = 5000 + 512, but no output carries 512, so the k:1 exchange has no subset to
        // swap and contributes nothing; the three 5000s still contribute 3! on their own.
        assert_eq!(
            radix_mappings(&[5000, 5000, 5000, 5512], 6),
            Ambiguity::Diagnostic(6)
        );
        // With a 512 output present the same decomposition does contribute k × m! = 2 × 1!.
        assert_eq!(
            radix_mappings(&[512, 5000, 5512], 6),
            Ambiguity::Diagnostic(1 + 1 + 2)
        );
    }

    #[test]
    fn radix_mappings_is_not_a_bound_on_the_mapping_count() {
        // One input funding three equal outputs admits exactly one sub-transaction mapping, and
        // the call cannot see that: it does not take the inputs. Tagging it `Diagnostic` is what
        // keeps a consumer from reading 6 as a floor on 1.
        let counted = radix_mappings(&[5000, 5000, 5000], 6);
        assert_eq!(counted, Ambiguity::Diagnostic(6));
        assert_eq!(counted.lower_bound_count(), None);
        assert!(!counted.is_exact());
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
        let outputs: Vec<u64> = std::iter::repeat_n(131_072u64, 20)
            .chain(std::iter::repeat_n(262_144u64, 20))
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

        /// The composed invariant, over a width where the convolution can actually truncate:
        /// `w_sparse` must never report more than the enumerated truth, whatever bound it carries.
        #[test]
        fn w_sparse_le_w_brute(
            inputs in prop::collection::vec(1u64..=30, 2..=12),
            outputs in prop::collection::vec(1u64..=30, 1..=6),
        ) {
            let brute = w_brute(&inputs, &outputs, inputs.len());
            let sparse = w_sparse(&inputs, &outputs, inputs.len(), nz(1_000_000));
            prop_assume!(!brute.is_unknown() && !sparse.is_unknown());
            let brute_n = brute.lower_bound_count().unwrap_or(0);
            let sparse_n = sparse.lower_bound_count().unwrap_or(0);
            prop_assert!(sparse_n <= brute_n, "sparse {} > brute {}", sparse_n, brute_n);
            if sparse.is_exact() {
                prop_assert_eq!(sparse_n, brute_n);
            }
        }

        #[test]
        fn radix_mappings_monotonic_in_k(
            outputs in prop::collection::vec(1u64..=1_000_000, 1..=4),
            k1 in 1usize..=6,
            k2 in 1usize..=6,
        ) {
            let (low, high) = (k1.min(k2), k1.max(k2));
            let m_low = radix_mappings(&outputs, low).count().unwrap_or(0);
            let m_high = radix_mappings(&outputs, high).count().unwrap_or(0);
            prop_assert!(m_high >= m_low);
        }

        #[test]
        fn radix_mappings_always_diagnostic(
            outputs in prop::collection::vec(1u64..=1_000_000, 1..=4),
            k in 1usize..=6,
        ) {
            let counted = radix_mappings(&outputs, k);
            prop_assert!(matches!(counted, Ambiguity::Diagnostic(_)));
            prop_assert_eq!(counted.lower_bound_count(), None);
        }

        /// Duplicating the output multiset multiplies each value's multiplicity but adds no
        /// distinct value, so the count moves only through `m!` — never through a per-coin
        /// repetition of the same summand.
        #[test]
        fn radix_mappings_counts_values_not_coins(
            outputs in prop::collection::vec(1u64..=1_000_000, 1..=4),
            k in 1usize..=6,
        ) {
            let mut permuted = outputs.clone();
            permuted.reverse();
            prop_assert_eq!(radix_mappings(&outputs, k), radix_mappings(&permuted, k));

            let distinct: std::collections::BTreeSet<u64> = outputs.iter().copied().collect();
            if distinct.len() == outputs.len() {
                // No repeats: every m is 1, so every surviving summand is k × 1! = k, and the
                // total is bounded by k per distinct value.
                let total = radix_mappings(&outputs, k).count().unwrap_or(0);
                prop_assert!(total <= (k as u128) * distinct.len() as u128);
            }
        }

        #[test]
        fn radix_mappings_zero_max_size_is_zero(
            outputs in prop::collection::vec(1u64..=100, 1..=4),
        ) {
            prop_assert_eq!(radix_mappings(&outputs, 0), Ambiguity::Diagnostic(0));
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

    mod w_count_tests {
        use super::*;

        /// `ln C(n, k)` via direct log-space summation (exact enough for f64 comparison;
        /// no crate dependency needed for this one test fixture).
        fn ln_binomial(n: u64, k: u64) -> f64 {
            (1..=k).map(|i| ((n - k + i) as f64 / i as f64).ln()).sum()
        }

        #[test]
        fn w_count_tiny_uses_brute() {
            let report = w_count(&[500, 300], &[500, 300]);
            assert_eq!(report.method, Method::Brute);
            assert_eq!(report.ambiguity, Ambiguity::Exact(2));
        }

        /// N=25 > BRUTE_MAX rules out brute, but the reachable sum is tiny (all-ones
        /// inputs), so the DP tier resolves it exactly.
        #[test]
        fn w_count_mid_n_small_sum_uses_dp() {
            let inputs = vec![1u64; 25];
            let outputs = vec![3u64, 4];
            let report = w_count(&inputs, &outputs);
            assert_eq!(report.method, Method::Dp);
            assert!(report.ambiguity.is_exact());
            // Cross-check against w_dp called directly with the same budget.
            let direct = w_dp(&inputs, &outputs, inputs.len(), DP_MAX_CELLS);
            assert_eq!(report.ambiguity, direct);
        }

        /// N=25 distinct near-coprime, astronomically large inputs (gcd 1, ΣA ≈ 3.25e9):
        /// the DP tier's cell budget (proportional to ΣA) overflows, but sparse
        /// convolution's cost tracks output support, not input magnitude, so it still
        /// resolves the single-target instance.
        #[test]
        fn w_count_dp_overflow_uses_sparse() {
            let inputs: Vec<u64> = (1..=25u64).map(|i| i * 10_000_000 + 1).collect();
            let target = inputs[0] + inputs[1];
            let outputs = vec![target];
            assert_eq!(
                w_dp(&inputs, &outputs, inputs.len(), DP_MAX_CELLS),
                Ambiguity::Unknown,
                "fixture must actually overflow the DP tier for this test to be meaningful"
            );
            let report = w_count(&inputs, &outputs);
            assert_eq!(report.method, Method::Sparse);
            assert!(!report.ambiguity.is_unknown());
        }

        /// Outputs with no denomination structure (distinct powers of two) blow past
        /// `output_subsums`'s reachable-sum budget — every tier gates on that same
        /// helper, so all four fall through to `Unknown`.
        #[test]
        fn w_count_intractable_output_subsums_is_none() {
            let inputs: Vec<u64> = (1..=30u64).map(|i| i * 999_999_937 + 7).collect();
            let outputs: Vec<u64> = (1..=40u64).map(|i| 1u64 << i).collect();
            assert!(output_subsums(&outputs).is_none());
            let report = w_count(&inputs, &outputs);
            assert_eq!(report.method, Method::None);
            assert_eq!(report.ambiguity, Ambiguity::Unknown);
        }

        /// Sasamoto's Dense-regime instance from `density_regime::regime_dense_high_n`
        /// (N=100 equal coins, E = MAX_MONEY/4). Cross-validates `w_count`'s result
        /// against `ln C(100, 25)` regardless of which tier resolves it.
        ///
        /// NOTE: with equal-valued inputs, gcd-normalization collapses the DP tier to
        /// trivial size (100 cells of value 1 each), so `w_dp` computes this exactly and
        /// wins the cascade before `w_sparse`/`w_sasamoto` are even tried — `Method::Dp`,
        /// not `Method::Sasamoto`, is what actually fires here. More fundamentally,
        /// `w_sparse` and `w_sasamoto` share the exact same gate (`output_subsums`
        /// succeeding), `w_sparse` is tried first, and `w_sparse` never itself resolves
        /// to `Unknown` once that gate passes (its `Bound` is always `Exact` or
        /// `LowerBound`) — so in this cascade `Method::Sasamoto` can only fire when
        /// `output_subsums` fails, but that also disables `w_sasamoto` itself (it gates
        /// on the same helper). `Method::Sasamoto` is therefore unreachable via this
        /// exact cascade; the assertion below is written to hold regardless of which
        /// tier answers, so it stays meaningful if the primitives' internals change.
        #[test]
        fn w_count_sasamoto_matches_binomial() {
            let n: usize = 100;
            let c: u64 = 21_000_000 * 100_000_000 / n as u64;
            let inputs = vec![c; n];
            let outputs = vec![25 * c];

            // The Sasamoto primitive itself, in isolation, is accurate in-regime.
            let target = outputs[0];
            let log_w_direct =
                sasamoto_approx(&inputs, target).expect("N=100 equal coins at E=ΣA/4 is Dense");
            let ln_c = ln_binomial(100, 25);
            let rel_err_direct = (log_w_direct - ln_c).abs() / ln_c.abs();
            assert!(
                rel_err_direct < 1e-3,
                "sasamoto_approx alone: log_w={log_w_direct}, ln C(100,25)={ln_c}, rel_err={rel_err_direct}"
            );

            // Whichever tier w_count actually lands on for this instance must agree.
            let report = w_count(&inputs, &outputs);
            assert!(
                !report.ambiguity.is_unknown(),
                "instance is Dense; some tier must resolve it"
            );
            let log_w = report
                .ambiguity
                .log()
                .expect("non-Unknown Ambiguity always has a log()");
            let rel_err = (log_w - ln_c).abs() / ln_c.abs();
            assert!(
                rel_err < 1e-3,
                "w_count (method={:?}): log_w={log_w}, ln C(100,25)={ln_c}, rel_err={rel_err}",
                report.method,
            );
        }

        #[test]
        fn w_count_lands_on_sasamoto_for_saturated_dense() {
            // The large-dense-coinjoin case exact counting saturates on. Consecutive values -> gcd 1
            // (DP can't collapse; the huge normalized sum overflows DP_MAX_CELLS -> Unknown); a
            // target ~ΣA/4 (x ≈ 1/4, well within the saddle-point solver domain) is reached by an
            // astronomical number of subsets, so the sparse counter saturates -> LowerBound (post the
            // count-saturation fix). Only the Dense saddle-point resolves it, so w_count returns
            // Method::Sasamoto with the accurate log-magnitude instead of the saturated LowerBound.
            let base: u64 = 21_000_000 * 100_000_000 / 400; // ~MAX_MONEY/400, Dense per prior probe
            let inputs: Vec<u64> = (0..100u64).map(|i| base + i).collect();
            let sum_a: u128 = inputs.iter().map(|&x| u128::from(x)).sum();
            let outputs = vec![(sum_a / 4) as u64]; // x ≈ 1/4: κ_c peak, saddle-point well-defined

            let report = w_count(&inputs, &outputs);
            assert_eq!(
                report.method,
                Method::Sasamoto,
                "expected Sasamoto tier; got {:?} with {:?}",
                report.method,
                report.ambiguity
            );
            assert!(report.ambiguity.is_approx());
            assert!(report.ambiguity.log().is_some_and(f64::is_finite));
        }

        proptest! {
            /// On small instances (N<=8, small values) w_brute/w_dp/w_sparse always agree
            /// exactly — so whichever tier w_count lands on, the count is the same.
            #[test]
            fn w_count_exact_methods_agree(
                inputs in prop::collection::vec(1u64..=50, 1..=8),
                outputs in prop::collection::vec(1u64..=50, 1..=6),
            ) {
                let ms = inputs.len();
                let brute = w_brute(&inputs, &outputs, ms);
                let dp = w_dp(&inputs, &outputs, ms, DP_MAX_CELLS);
                let sparse = w_sparse(&inputs, &outputs, ms, DEFAULT_MEMORY_BUDGET);
                prop_assert_eq!(brute, dp);
                prop_assert_eq!(dp, sparse);
                prop_assert!(brute.is_exact());
            }

            /// Guarantee ordering: on the same small/exact-agreeing fixtures, w_count's
            /// result is always Exact — never a LowerBound/LogApprox downgrade, because
            /// an exact tier always resolves them first.
            #[test]
            fn w_count_never_downgrades(
                inputs in prop::collection::vec(1u64..=50, 1..=8),
                outputs in prop::collection::vec(1u64..=50, 1..=6),
            ) {
                let report = w_count(&inputs, &outputs);
                prop_assert!(report.ambiguity.is_exact());
                prop_assert!(matches!(report.method, Method::Brute | Method::Dp | Method::Sparse));
            }
        }
    }
}
