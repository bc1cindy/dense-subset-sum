use super::budget::{DEFAULT_CONV_SEED, GradedSumsetBudget, dispatch_sparse};
use super::types::{Bound, Count, classify};
use crate::count::sparse_conv::{Field, Goldilocks, SplitMix};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::OnceLock;

/// `degrees[d]: sum → count` for subsets of size `d`. `W(E) = count_total(e)`; `W(M, E) = count_at(m, e)`.
pub struct GradedSumset<F: Field = Goldilocks> {
    degrees: Vec<HashMap<u64, u32>>,
    max_degree: usize,
    n_inputs: usize,
    bound: Bound,
    blowup_level: Option<usize>,
    depth: usize,
    /// Lazily-populated dedup of `degrees[*].keys()`; invalidated on any bucket mutation.
    sums_cache: OnceLock<Vec<u64>>,
    _field: PhantomData<F>,
}

impl<F: Field> GradedSumset<F> {
    /// Empty sumset (no inputs); represents `{0:1}` at degree 0 only.
    #[must_use]
    pub fn empty() -> Self {
        Self::singleton(0, 0)
    }

    /// Subsets of size ≤ `max_degree`. `targets` are pinned past `cap_to_top` truncation.
    ///
    /// ```
    /// use dense_subset_sum::GradedSumset;
    /// let s: GradedSumset = GradedSumset::bounded(&[1u64, 2, 3, 4, 5], &[5], 5);
    /// assert_eq!(s.count_total(5).visible(), 3); // {5}, {1,4}, {2,3}
    /// ```
    #[must_use]
    pub fn bounded(elements: &[u64], targets: &[u64], max_degree: usize) -> Self {
        Self::builder(elements, GradedSumsetBudget::<F>::default(), targets).bounded(max_degree)
    }

    /// Pinning `targets` protects them from silent `cap_to_top` zeroing; `&[]` to skip.
    #[must_use]
    pub fn builder<'a>(
        elements: &'a [u64],
        budget: GradedSumsetBudget<F>,
        targets: &'a [u64],
    ) -> GradedSumsetBuilder<'a, F> {
        GradedSumsetBuilder {
            elements,
            budget,
            pinned: targets,
        }
    }

    /// W(M, E): subsets of size exactly `m` summing to `e`.
    #[must_use]
    pub fn count_at(&self, m: usize, e: u64) -> Count {
        let raw = self
            .degrees
            .get(m)
            .and_then(|bucket| bucket.get(&e).copied());
        let truncated = self.bound_at(m) == Bound::LowerBound;
        match (raw, truncated) {
            (Some(n), false) => Count::Confirmed(n),
            (Some(n), true) => Count::Truncated(n),
            (None, false) => Count::Absent,
            (None, true) => Count::Unknown,
        }
    }

    /// W(E) = Σ_m W(m, E); `LowerBound` if any degree truncated or `max_degree < n_inputs`.
    #[must_use]
    pub fn count_total(&self, e: u64) -> Count {
        let mut sum = 0u32;
        for m in 0..=self.max_degree {
            if let Some(bucket) = self.degrees.get(m) {
                if let Some(&v) = bucket.get(&e) {
                    sum = sum.saturating_add(v);
                }
            }
        }
        let truncated = self.bound_total() == Bound::LowerBound;
        classify(sum, truncated)
    }

    /// Σ_{m ∈ range} W(m, E).
    #[must_use]
    pub fn count_in_range(&self, range: std::ops::RangeInclusive<usize>, e: u64) -> Count {
        let lo = *range.start();
        let hi = (*range.end()).min(self.max_degree);
        if lo > hi {
            return Count::Absent;
        }
        let mut sum = 0u32;
        for m in lo..=hi {
            if let Some(bucket) = self.degrees.get(m) {
                if let Some(&v) = bucket.get(&e) {
                    sum = sum.saturating_add(v);
                }
            }
        }
        let truncated = self.bound == Bound::LowerBound;
        classify(sum, truncated)
    }

    /// `Exact` if `m > max_degree` (count is trivially 0).
    #[must_use]
    pub fn bound_at(&self, m: usize) -> Bound {
        if m > self.max_degree {
            Bound::Exact
        } else {
            self.bound
        }
    }

    /// `LowerBound` if `max_degree < n_inputs` (degrees not stored).
    #[must_use]
    pub fn bound_total(&self) -> Bound {
        if self.max_degree < self.n_inputs {
            Bound::LowerBound
        } else {
            self.bound
        }
    }

    /// Caches dedup on first call; subsequent calls reuse.
    pub fn sums_total(&self) -> impl Iterator<Item = u64> + '_ {
        self.sums_cache
            .get_or_init(|| {
                let mut seen = std::collections::HashSet::new();
                self.degrees
                    .iter()
                    .flat_map(|bucket| bucket.keys().copied())
                    .filter(|s| seen.insert(*s))
                    .collect()
            })
            .iter()
            .copied()
    }

    pub fn sums_at(&self, m: usize) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.degrees
            .get(m)
            .into_iter()
            .flat_map(|l| l.iter().map(|(&k, &v)| (k, v)))
    }

    #[must_use]
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[must_use]
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// `true` iff no degree truncation.
    #[must_use]
    pub fn is_exhaustive(&self) -> bool {
        self.max_degree >= self.n_inputs
    }

    /// Earliest merge level where support exceeded `max_size`.
    #[must_use]
    pub fn blowup_level(&self) -> Option<usize> {
        self.blowup_level
    }

    /// Σ bucket sizes across all degrees.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.degrees.iter().map(HashMap::len).sum()
    }

    /// Any count reached `u32::MAX`.
    #[must_use]
    pub fn saturated(&self) -> bool {
        self.degrees
            .iter()
            .any(|bucket| bucket.values().any(|&v| v == u32::MAX))
    }

    #[cfg(test)]
    pub(crate) fn convolve(a: &Self, b: &Self) -> Self {
        Self::convolve_with(a, b, &GradedSumsetBudget::<F>::default())
    }

    #[cfg(test)]
    pub(crate) fn convolve_with(a: &Self, b: &Self, budget: &GradedSumsetBudget<F>) -> Self {
        a.convolve_degrees(b, budget, DEFAULT_CONV_SEED)
    }

    /// Every sum in `self` reachable by ≥ threshold subsets in `other`.
    #[must_use]
    pub fn covers(&self, other: &Self, threshold: u32) -> bool {
        self.sums_total()
            .all(|s| other.count_total(s).visible() >= threshold)
    }

    /// Any pair `(s_self, s_other)` with `s_self − s_other = target`.
    #[must_use]
    pub fn includes_balance(&self, other: &Self, target: i64) -> bool {
        let (smaller, larger, t) = if self.total_len() <= other.total_len() {
            (self, other, target)
        } else {
            (other, self, -target)
        };
        smaller.sums_total().any(|s| {
            let Ok(s_signed) = i64::try_from(s) else {
                return false;
            };
            let Some(complement) = s_signed.checked_sub(t) else {
                return false;
            };
            u64::try_from(complement).is_ok_and(|c| larger.count_total(c).visible() > 0)
        })
    }

    /// Saturating count of pairs `(s_self, s_other)` with `s_self − s_other = target`.
    #[must_use]
    pub fn count_balance(&self, other: &Self, target: i64) -> u32 {
        self.sums_total()
            .filter_map(|s| {
                let s_signed = i64::try_from(s).ok()?;
                let needed = s_signed.checked_sub(target)?;
                let needed_u64 = u64::try_from(needed).ok()?;
                let count = self
                    .count_total(s)
                    .visible()
                    .saturating_mul(other.count_total(needed_u64).visible());
                Some(count)
            })
            .fold(0u32, u32::saturating_add)
    }

    fn singleton(max_degree: usize, n_inputs: usize) -> Self {
        let mut degrees = vec![HashMap::new(); max_degree + 1];
        degrees[0].insert(0u64, 1u32);
        Self {
            degrees,
            max_degree,
            n_inputs,
            bound: Bound::Exact,
            blowup_level: None,
            depth: 0,
            sums_cache: OnceLock::new(),
            _field: PhantomData,
        }
    }

    fn from_element(a: u64, max_degree: usize) -> Self {
        let mut degrees = vec![HashMap::new(); max_degree + 1];
        degrees[0].insert(0u64, 1u32);
        if max_degree >= 1 {
            *degrees[1].entry(a).or_insert(0) += 1;
        }
        Self {
            degrees,
            max_degree,
            n_inputs: 1,
            bound: Bound::Exact,
            blowup_level: None,
            depth: 0,
            sums_cache: OnceLock::new(),
            _field: PhantomData,
        }
    }

    /// Cross-product `(d_a, d_b) → d_a + d_b`; `sparse_conv` above [`SPARSE_CONV_CROSSOVER`], direct below.
    fn convolve_degrees(&self, other: &Self, budget: &GradedSumsetBudget<F>, seed: u64) -> Self {
        debug_assert_eq!(self.max_degree, other.max_degree);
        let max_d = self.max_degree;
        let mut result_degrees: Vec<HashMap<u64, u32>> = vec![HashMap::new(); max_d + 1];
        let mut local_seed = seed;
        let mut sub_conv_truncated = false;

        for d_a in 0..=max_d {
            if self.degrees[d_a].is_empty() {
                continue;
            }
            for d_b in 0..=(max_d - d_a) {
                if other.degrees[d_b].is_empty() {
                    continue;
                }
                let total = d_a + d_b;
                let bucket_a = &self.degrees[d_a];
                let bucket_b = &other.degrees[d_b];
                let product = bucket_a.len().saturating_mul(bucket_b.len());

                if product < SPARSE_CONV_CROSSOVER {
                    for (&ka, &va) in bucket_a {
                        for (&kb, &vb) in bucket_b {
                            let entry = result_degrees[total].entry(ka + kb).or_insert(0);
                            *entry = entry.saturating_add(va.saturating_mul(vb));
                        }
                    }
                    continue;
                }

                let av: Vec<(u64, u64)> =
                    bucket_a.iter().map(|(&k, &v)| (k, u64::from(v))).collect();
                let bv: Vec<(u64, u64)> =
                    bucket_b.iter().map(|(&k, &v)| (k, u64::from(v))).collect();
                let conv = dispatch_sparse::<F>(
                    &av,
                    &bv,
                    local_seed,
                    budget.max_inner_iters(),
                    budget.las_vegas(),
                );
                local_seed = local_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
                if conv.termination() == crate::count::sparse_conv::Termination::LowerBound {
                    sub_conv_truncated = true;
                }
                for (sum, c) in conv.into_support() {
                    let entry = result_degrees[total].entry(sum).or_insert(0);
                    let c32 = u32::try_from(c).unwrap_or(u32::MAX);
                    *entry = entry.saturating_add(c32);
                }
            }
        }

        let blowup_level = match (self.blowup_level, other.blowup_level) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (s, o) => s.or(o),
        };
        let bound = if sub_conv_truncated {
            Bound::LowerBound
        } else {
            self.bound.join(other.bound)
        };
        Self {
            degrees: result_degrees,
            max_degree: max_d,
            n_inputs: self.n_inputs + other.n_inputs,
            bound,
            blowup_level,
            depth: self.depth.max(other.depth) + 1,
            sums_cache: OnceLock::new(),
            _field: PhantomData,
        }
    }

    /// Each bucket keeps its proportional slice of `max_size`; pinned sums survive.
    fn cap_to_top<P: Fn(u64) -> bool>(&mut self, max_size: usize, is_pinned: P) {
        let degree_count = self.degrees.len();
        if degree_count == 0 {
            return;
        }
        let per_bucket = max_size.div_ceil(degree_count);
        for bucket in &mut self.degrees {
            cap_to_top_counts(bucket, per_bucket, &is_pinned);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GradedSumsetBuilder<'a, F: Field = Goldilocks> {
    elements: &'a [u64],
    budget: GradedSumsetBudget<F>,
    pinned: &'a [u64],
}

impl<F: Field> GradedSumsetBuilder<'_, F> {
    #[must_use]
    pub fn bounded(self, max_degree: usize) -> GradedSumset<F> {
        bounded_merge::<F>(self.elements, max_degree, &self.budget, self.pinned)
    }
}

/// `|A|·|B|` threshold below which direct loop beats sparse_conv (NTT setup dominates).
const SPARSE_CONV_CROSSOVER: usize = 1 << 20;

struct BoundedMergeState<'a, F: Field> {
    budget: &'a GradedSumsetBudget<F>,
    max_size: usize,
    pinned: std::collections::HashSet<u64>,
    has_pins: bool,
    overflowed: bool,
    rng: SplitMix,
    _field: PhantomData<F>,
}

impl<'a, F: Field> BoundedMergeState<'a, F> {
    fn new(budget: &'a GradedSumsetBudget<F>, pinned_sums: &[u64]) -> Self {
        let pinned: std::collections::HashSet<u64> = pinned_sums.iter().copied().collect();
        let has_pins = !pinned.is_empty();
        Self {
            budget,
            max_size: budget.max_size(),
            pinned,
            has_pins,
            overflowed: false,
            rng: SplitMix::new(DEFAULT_CONV_SEED),
            _field: PhantomData,
        }
    }

    /// Three `LowerBound` triggers: pre-merge product skip, sub-conv truncation, post-merge `cap_to_top`.
    fn merge_pair_with_cap_check(
        &mut self,
        left: GradedSumset<F>,
        right: GradedSumset<F>,
    ) -> GradedSumset<F> {
        if !self.has_pins && left.total_len().saturating_mul(right.total_len()) > self.max_size {
            self.overflowed = true;
            let level = left.depth.max(right.depth) + 1;
            let mut survivor = if left.total_len() >= right.total_len() {
                left
            } else {
                right
            };
            survivor.depth = level;
            survivor.blowup_level.get_or_insert(level);
            survivor.bound = Bound::LowerBound;
            return survivor;
        }
        let seed = self.rng.next();
        let mut merged = left.convolve_degrees(&right, self.budget, seed);
        if merged.bound == Bound::LowerBound {
            self.overflowed = true;
        }
        if merged.total_len() > self.max_size {
            self.overflowed = true;
            merged.blowup_level.get_or_insert(merged.depth);
            let pinned = &self.pinned;
            merged.cap_to_top(self.max_size, |s| pinned.contains(&s));
            merged.bound = Bound::LowerBound;
        }
        merged
    }
}

fn bounded_merge<F: Field>(
    elements: &[u64],
    max_degree: usize,
    budget: &GradedSumsetBudget<F>,
    pinned_sums: &[u64],
) -> GradedSumset<F> {
    let n = elements.len();
    if max_degree == 0 || elements.is_empty() {
        return GradedSumset::<F>::singleton(max_degree, n);
    }

    let mut state = BoundedMergeState::<F>::new(budget, pinned_sums);
    let leaf = |a: u64| GradedSumset::<F>::from_element(a, max_degree);
    let mut merged = merge_tree(
        elements,
        GradedSumset::<F>::singleton(max_degree, 0),
        leaf,
        |left, right| state.merge_pair_with_cap_check(left, right),
    );

    // Degree truncation is reported via bound_total()/is_exhaustive(); bound flips only on support truncation.
    if state.overflowed {
        merged.bound = Bound::LowerBound;
    }
    merged.n_inputs = n;
    merged
}

/// Sorted leaves fold pairwise upward; close-valued neighbours coalesce earlier, shrinking intermediate support.
fn merge_tree<T, L, C>(elements: &[u64], empty: T, leaf: L, mut combine: C) -> T
where
    L: Fn(u64) -> T,
    C: FnMut(T, T) -> T,
{
    if elements.is_empty() {
        return empty;
    }
    let mut sorted: Vec<u64> = elements.to_vec();
    sorted.sort_unstable();
    let mut layer: Vec<T> = sorted.into_iter().map(&leaf).collect();
    while layer.len() > 1 {
        let mut next: Vec<T> = Vec::with_capacity(layer.len().div_ceil(2));
        let mut drain = layer.into_iter();
        while let Some(a) = drain.next() {
            next.push(match drain.next() {
                Some(b) => combine(a, b),
                None => a,
            });
        }
        layer = next;
    }
    layer
        .into_iter()
        .next()
        .expect("non-empty layer collapses to one")
}

/// Preserves idx=0 (count(0)=2^k invariant), pinned sums, and top frequencies.
fn cap_to_top_counts<F: Fn(u64) -> bool>(h: &mut HashMap<u64, u32>, max_size: usize, is_pinned: F) {
    if h.len() <= max_size {
        return;
    }
    let zero_count = h.remove(&0);
    let entries: Vec<(u64, u32)> = h.drain().collect();
    let (pinned, mut rest): (Vec<_>, Vec<_>) =
        entries.into_iter().partition(|(k, _)| is_pinned(*k));
    let reserved = usize::from(zero_count.is_some());
    let remaining = max_size.saturating_sub(reserved + pinned.len());
    rest.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    rest.truncate(remaining);
    if let Some(c) = zero_count {
        h.insert(0, c);
    }
    for (k, v) in pinned {
        h.insert(k, v);
    }
    for (k, v) in rest {
        h.insert(k, v);
    }
}

#[cfg(test)]
#[path = "graded_tests.rs"]
mod tests;
