//! Exact subset-sum counts: W(E) and W(M, E). 0/1 DP plus a sparse-convolution
//! sumset over a divide-and-conquer tree, with [`brute_force_w`] /
//! [`brute_force_w_restricted`] as test oracles. `Truncated` variants are lower
//! bounds returned when `max_entries` is reached or `u128` overflows. W(M, E) mirrors W(E) in
//! API shape; only W(E) has a dense variant, since `(m, sum)` cells scale as
//! `O((N+1)·sum)`.

use std::collections::{HashMap, hash_map};

use crate::sasamoto::gcd_slice;

/// Default memory budget: 2^26 entries ≈ 2GB.
pub const DEFAULT_MAX_ENTRIES: usize = 67_108_864;

/// Size of one sumset entry: `u64` sum + `u128` count.
const ENTRY_SIZE_BYTES: usize = std::mem::size_of::<(u64, u128)>();

/// Convolution tree leaf size; leaves enumerate 2^12 = 4096 subsets directly.
const LEAF_SIZE: usize = 12;

/// Sparse → dense representation cutoff.
const DENSE_THRESHOLD: f64 = 0.5;

/// Brute-force enumeration cap: 2^25 ≈ 33M masks.
const MAX_BRUTE_FORCE_N: usize = 25;

/// Lookup tier memory budget.
#[derive(Debug, Clone)]
pub struct LookupConfig {
    pub max_entries: usize,
}

impl Default for LookupConfig {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }
}

impl LookupConfig {
    pub fn from_memory_bytes(bytes: usize) -> Self {
        Self {
            max_entries: bytes / ENTRY_SIZE_BYTES,
        }
    }

    pub fn from_max_entries(n: usize) -> Self {
        Self { max_entries: n }
    }

    pub fn memory_bytes(&self) -> usize {
        self.max_entries * ENTRY_SIZE_BYTES
    }
}

/// Fisher-Yates seeded by input content: same multiset ⇒ same shuffle.
fn shuffle_deterministic(set: &mut [u64]) {
    let mut seed: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for &v in set.iter() {
        seed = seed.wrapping_mul(0x100000001b3) ^ v; // FNV-1a prime
    }
    let mut state = seed | 1;
    for i in (1..set.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005) // PCG multiplier
            .wrapping_add(1442695040888963407); // PCG increment
        let j = ((state >> 33) as usize) % (i + 1);
        set.swap(i, j);
    }
}

// Unrestricted W(E)

/// Enumerates 2^N subsets; `None` for `N > MAX_BRUTE_FORCE_N` or `u64` overflow.
pub fn brute_force_w(original_set: &[u64], e_target: u64) -> Option<u128> {
    let n = original_set.len();
    if n > MAX_BRUTE_FORCE_N {
        return None;
    }
    original_set
        .iter()
        .copied()
        .try_fold(0u64, u64::checked_add)?;
    let mut count: u128 = 0;
    for mask in 0..(1u64 << n) {
        let sum: u64 = (0..n)
            .filter(|&j| mask & (1 << j) != 0)
            .map(|j| original_set[j])
            .sum();
        if sum == e_target {
            count += 1;
        }
    }
    Some(count)
}

/// Lower-bound sumset; distinct from `Sparse` to keep truncation provenance.
#[derive(Debug, Clone)]
pub struct LowerBound(HashMap<u64, u128>);

impl LowerBound {
    pub fn get(&self, e: u64) -> u128 {
        self.0.get(&e).copied().unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn iter(&self) -> hash_map::Iter<'_, u64, u128> {
        self.0.iter()
    }

    fn into_inner(self) -> HashMap<u64, u128> {
        self.0
    }
}

/// Dense sumset: `counts[i]` is the count for `offset + i`; build via [`Sumset::dense`].
#[derive(Debug, Clone)]
pub struct DenseSumset {
    /// Smallest sum represented (i.e. the index-0 sum).
    offset: u64,
    /// Counts indexed by `sum - offset`.
    counts: Vec<u128>,
    /// Number of positive entries in `counts`, cached for O(1) `len()`.
    nonzero_count: usize,
}

impl DenseSumset {
    fn new(offset: u64, counts: Vec<u128>) -> Self {
        let nonzero_count = counts.iter().filter(|&&c| c > 0).count();
        Self {
            offset,
            counts,
            nonzero_count,
        }
    }

    pub fn get(&self, e: u64) -> u128 {
        e.checked_sub(self.offset)
            .and_then(|d| usize::try_from(d).ok())
            .and_then(|i| self.counts.get(i).copied())
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.nonzero_count
    }

    pub fn is_empty(&self) -> bool {
        self.nonzero_count == 0
    }

    pub fn max_sum(&self) -> Option<u64> {
        self.counts
            .iter()
            .enumerate()
            .rfind(|&(_, &c)| c > 0)
            .map(|(i, _)| self.offset + i as u64)
    }

    /// Iterates `(sum, count)` over positive-count entries.
    pub fn iter(&self) -> DenseSumsetIter<'_> {
        DenseSumsetIter {
            offset: self.offset,
            inner: self.counts.iter().enumerate(),
        }
    }
}

/// Skips zero entries; yields `(offset + i, count)` for positive `count`.
pub struct DenseSumsetIter<'a> {
    offset: u64,
    inner: std::iter::Enumerate<std::slice::Iter<'a, u128>>,
}

impl Iterator for DenseSumsetIter<'_> {
    type Item = (u64, u128);

    fn next(&mut self) -> Option<Self::Item> {
        for (i, &c) in self.inner.by_ref() {
            if c > 0 {
                return Some((self.offset + i as u64, c));
            }
        }
        None
    }
}

/// Sumset of A: counts of subsets per reachable sum.
#[derive(Debug, Clone)]
pub enum Sumset {
    /// Complete sumset via HashMap; every count > 0.
    Sparse(HashMap<u64, u128>),
    /// Complete sumset via [`DenseSumset`] (Vec-backed, compact).
    Dense(DenseSumset),
    /// Lower bound: memory budget reached during convolution.
    Truncated(LowerBound),
}

impl Sumset {
    /// Only public path to `Dense`; computes `nonzero_count` to preserve the invariant.
    pub fn dense(offset: u64, counts: Vec<u128>) -> Self {
        Self::Dense(DenseSumset::new(offset, counts))
    }

    /// Count of subsets summing to `e`; `0` if absent.
    pub fn get(&self, e: u64) -> u128 {
        match self {
            Self::Sparse(m) => m.get(&e).copied().unwrap_or(0),
            Self::Dense(d) => d.get(e),
            Self::Truncated(t) => t.get(e),
        }
    }

    /// Number of distinct reachable sums with positive count. O(1).
    pub fn len(&self) -> usize {
        match self {
            Self::Sparse(m) => m.len(),
            Self::Dense(d) => d.len(),
            Self::Truncated(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Largest reachable sum, or `None` for `Truncated` or empty.
    pub fn max_sum(&self) -> Option<u64> {
        match self {
            Self::Sparse(m) => m.keys().copied().max(),
            Self::Dense(d) => d.max_sum(),
            Self::Truncated(_) => None,
        }
    }

    /// Iterates `(sum, count)` over reachable subsets.
    pub fn iter(&self) -> SumsetIter<'_> {
        match self {
            Self::Sparse(m) => SumsetIter::Map(m.iter()),
            Self::Dense(d) => SumsetIter::Dense(d.iter()),
            Self::Truncated(t) => SumsetIter::Map(t.iter()),
        }
    }
}

/// Static-dispatch enum-iter for [`Sumset`]; avoids `Box<dyn Iterator>`.
pub enum SumsetIter<'a> {
    Map(hash_map::Iter<'a, u64, u128>),
    Dense(DenseSumsetIter<'a>),
}

impl Iterator for SumsetIter<'_> {
    type Item = (u64, u128);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Map(it) => it.next().map(|(&s, &c)| (s, c)),
            Self::Dense(it) => it.next(),
        }
    }
}

/// Subset-sum DP. Counts use `u128` and overflow silently for `N ≥ 128`.
pub fn dp_w(original_set: &[u64], e_target: u64, max_cells: usize) -> Option<u128> {
    let NormalizedDp {
        normalized,
        e_norm,
        sum_max,
    } = match normalize_for_dp(original_set, e_target) {
        DpInput::Ready(input) => input,
        DpInput::EarlyZero => return Some(0),
        DpInput::Degenerate => return None,
    };

    let sz = usize::try_from(sum_max).ok()?.checked_add(1)?;
    if sz > max_cells {
        return None;
    }

    let mut dp = vec![0u128; sz];
    dp[0] = 1;

    for &val in &normalized {
        // v == 0: `dp[j] += dp[j]` doubles each cell (zero freely in or out).
        let v = val as usize;
        for j in (v..sz).rev() {
            dp[j] += dp[j - v];
        }
    }

    Some(dp[e_norm as usize])
}

pub fn log_dp_w(original_set: &[u64], e_target: u64, max_cells: usize) -> Option<f64> {
    Some(log_count(dp_w(original_set, e_target, max_cells)?))
}

/// Post-gcd DP state. Constructed by [`normalize_for_dp`].
struct NormalizedDp {
    normalized: Vec<u64>,
    e_norm: u64,
    sum_max: u64,
}

/// Outcome of [`normalize_for_dp`].
enum DpInput {
    /// Inputs ready for the DP loop.
    Ready(NormalizedDp),
    /// E unreachable: not a multiple of gcd(A) or > Σa.
    EarlyZero,
    /// A degenerate (empty/all-zero) or Σa overflows u64.
    Degenerate,
}

/// gcd-normalizes A and E for the DP path.
fn normalize_for_dp(set: &[u64], e: u64) -> DpInput {
    let Some(g) = gcd_slice(set) else {
        return DpInput::Degenerate;
    };
    if !e.is_multiple_of(g) {
        return DpInput::EarlyZero;
    }
    let normalized: Vec<u64> = set.iter().map(|&v| v / g).collect();
    let e_norm = e / g;
    let Some(sum_max) = normalized.iter().copied().try_fold(0u64, u64::checked_add) else {
        return DpInput::Degenerate;
    };
    if e_norm > sum_max {
        return DpInput::EarlyZero;
    }
    DpInput::Ready(NormalizedDp {
        normalized,
        e_norm,
        sum_max,
    })
}

fn log_count<N: Into<u128>>(w: N) -> f64 {
    let w = w.into();
    if w == 0 {
        f64::NEG_INFINITY
    } else {
        (w as f64).ln()
    }
}

/// Full sumset of A. Input is deterministically shuffled to avoid ordering bias.
pub fn sumset(set: &[u64], cfg: &LookupConfig) -> Sumset {
    if set.is_empty() {
        let mut m = HashMap::new();
        m.insert(0u64, 1u128);
        return Sumset::Sparse(m);
    }
    let mut shuffled = set.to_vec();
    shuffle_deterministic(&mut shuffled);
    tree_sumset(&shuffled, cfg)
}

/// `W(E)` via [`sumset`]; lower bound if truncated.
pub fn lookup_w(set: &[u64], e_target: u64) -> Option<u128> {
    lookup_w_with_config(set, e_target, &LookupConfig::default())
}

/// Like [`lookup_w`], with explicit memory budget.
pub fn lookup_w_with_config(set: &[u64], e_target: u64, cfg: &LookupConfig) -> Option<u128> {
    if set.is_empty() {
        return None;
    }
    Some(sumset(set, cfg).get(e_target))
}

pub fn log_lookup_w(set: &[u64], e_target: u64) -> Option<f64> {
    log_lookup_w_with_config(set, e_target, &LookupConfig::default())
}

pub fn log_lookup_w_with_config(set: &[u64], e_target: u64, cfg: &LookupConfig) -> Option<f64> {
    Some(log_count(lookup_w_with_config(set, e_target, cfg)?))
}

/// Recursive divide-and-conquer over the multiset.
fn tree_sumset(set: &[u64], cfg: &LookupConfig) -> Sumset {
    if let Some(dense) = try_direct_dense(set, cfg) {
        return dense;
    }
    if set.len() <= LEAF_SIZE {
        return Sumset::Sparse(leaf_sumset(set));
    }
    let mid = set.len() / 2;
    let left = tree_sumset(&set[..mid], cfg);
    let right = tree_sumset(&set[mid..], cfg);
    convolve(left, right, cfg)
}

/// Leaf sumset of a small block: brute-force HashMap over 2^k subsets.
fn leaf_sumset(block: &[u64]) -> HashMap<u64, u128> {
    let n = block.len();
    let mut counts: HashMap<u64, u128> = HashMap::new();
    for mask in 0..(1u64 << n) {
        let sum: u64 = (0..n)
            .filter(|&j| mask & (1 << j) != 0)
            .map(|j| block[j])
            .sum();
        *counts.entry(sum).or_insert(0) += 1;
    }
    counts
}

/// If the subproblem is already dense, skip the tree and DP directly into a Vec.
fn try_direct_dense(set: &[u64], cfg: &LookupConfig) -> Option<Sumset> {
    let sz = dense_admission_sz(set, cfg, 1)?;
    let mut counts = vec![0u128; sz];
    counts[0] = 1;
    for &val in set {
        let v = val as usize;
        for j in (v..sz).rev() {
            counts[j] += counts[j - v];
        }
    }
    Some(Sumset::dense(0, counts))
}

/// Density gate; returns sumset width `sz` if dense and `rows × sz ≤ max_entries`.
fn dense_admission_sz(set: &[u64], cfg: &LookupConfig, rows: usize) -> Option<usize> {
    let sum: u64 = set.iter().copied().try_fold(0u64, u64::checked_add)?;
    if sum == 0 {
        return None;
    }
    let density = entries_upper_bound(set.len(), sum) as f64 / (sum as f64 + 1.0);
    if density <= DENSE_THRESHOLD {
        return None;
    }
    let sz = usize::try_from(sum).ok()?.checked_add(1)?;
    let cells = rows.checked_mul(sz)?;
    if cells > cfg.max_entries {
        return None;
    }
    Some(sz)
}

// Restricted W(M, E)

/// Enumerates 2^N (size, sum) pairs; same caps as [`brute_force_w`].
pub fn brute_force_w_restricted(original_set: &[u64], m: usize, e_target: u64) -> Option<u128> {
    let n = original_set.len();
    if n > MAX_BRUTE_FORCE_N {
        return None;
    }
    original_set
        .iter()
        .copied()
        .try_fold(0u64, u64::checked_add)?;
    let mut count: u128 = 0;
    for mask in 0..(1u64 << n) {
        if mask.count_ones() as usize != m {
            continue;
        }
        let sum: u64 = (0..n)
            .filter(|&j| mask & (1 << j) != 0)
            .map(|j| original_set[j])
            .sum();
        if sum == e_target {
            count += 1;
        }
    }
    Some(count)
}

/// Lower-bound restricted sumset; distinct from `Complete` to keep truncation provenance.
#[derive(Debug, Clone)]
pub struct RestrictedLowerBound(Vec<HashMap<u64, u128>>);

impl RestrictedLowerBound {
    pub fn get(&self, m: usize, e: u64) -> u128 {
        self.0
            .get(m)
            .and_then(|map| map.get(&e).copied())
            .unwrap_or(0)
    }

    pub fn num_sizes(&self) -> usize {
        self.0.len()
    }

    pub fn len(&self) -> usize {
        self.0.iter().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn slices(&self) -> &[HashMap<u64, u128>] {
        &self.0
    }

    fn into_inner(self) -> Vec<HashMap<u64, u128>> {
        self.0
    }
}

/// `W(M, E)`: count of size-`m` subsets of A summing to `e`.
#[derive(Debug, Clone)]
pub enum RestrictedSumset {
    /// Every count in every slice is exact.
    Complete(Vec<HashMap<u64, u128>>),
    /// Lower bound: at least one slice exceeded `max_entries` during convolution.
    Truncated(RestrictedLowerBound),
}

impl RestrictedSumset {
    /// Count of size-`m` subsets summing to `e`; `0` if absent.
    pub fn get(&self, m: usize, e: u64) -> u128 {
        match self {
            Self::Complete(slices) => slices
                .get(m)
                .and_then(|map| map.get(&e).copied())
                .unwrap_or(0),
            Self::Truncated(t) => t.get(m, e),
        }
    }

    /// Number of size slices (max representable size + 1).
    pub fn num_sizes(&self) -> usize {
        match self {
            Self::Complete(s) => s.len(),
            Self::Truncated(t) => t.num_sizes(),
        }
    }

    /// Total entries across all slices.
    pub fn len(&self) -> usize {
        match self {
            Self::Complete(s) => s.iter().map(|s| s.len()).sum(),
            Self::Truncated(t) => t.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Largest reachable sum across slices, or `None` for `Truncated` or empty.
    pub fn max_sum(&self) -> Option<u64> {
        match self {
            Self::Truncated(_) => None,
            Self::Complete(slices) => slices.iter().filter_map(|s| s.keys().copied().max()).max(),
        }
    }

    /// Iterates `(m, sum, count)` over reachable size-restricted subsets.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u64, u128)> + '_ {
        let slices: &[HashMap<u64, u128>] = match self {
            Self::Complete(v) => v,
            Self::Truncated(t) => t.slices(),
        };
        slices
            .iter()
            .enumerate()
            .flat_map(|(m, slice)| slice.iter().map(move |(&sum, &count)| (m, sum, count)))
    }
}

/// Subset-sum DP for size `m`; same returns and `N ≥ 128` caveat as [`dp_w`].
pub fn dp_w_restricted(
    original_set: &[u64],
    m: usize,
    e_target: u64,
    max_cells: usize,
) -> Option<u128> {
    if m > original_set.len() {
        return Some(0);
    }
    if m == 0 {
        return Some(u128::from(e_target == 0));
    }

    let NormalizedDp {
        normalized,
        e_norm,
        sum_max,
    } = match normalize_for_dp(original_set, e_target) {
        DpInput::Ready(input) => input,
        DpInput::EarlyZero => return Some(0),
        DpInput::Degenerate => return None,
    };

    let sz = usize::try_from(sum_max).ok()?.checked_add(1)?;
    let dp_rows = m + 1;
    let cells = dp_rows.checked_mul(sz)?;
    if cells > max_cells {
        return None;
    }

    let mut dp = vec![vec![0u128; sz]; dp_rows];
    dp[0][0] = 1;

    for &val in &normalized {
        let v = val as usize;
        for mm in (1..=m).rev() {
            for j in (v..sz).rev() {
                dp[mm][j] += dp[mm - 1][j - v];
            }
        }
    }

    Some(dp[m][e_norm as usize])
}

pub fn log_dp_w_restricted(
    original_set: &[u64],
    m: usize,
    e_target: u64,
    max_cells: usize,
) -> Option<f64> {
    Some(log_count(dp_w_restricted(
        original_set,
        m,
        e_target,
        max_cells,
    )?))
}

/// Full size-restricted sumset of A. Input is shuffled deterministically.
pub fn sumset_restricted(set: &[u64], cfg: &LookupConfig) -> RestrictedSumset {
    if set.is_empty() {
        let mut m0 = HashMap::new();
        m0.insert(0u64, 1u128);
        return RestrictedSumset::Complete(vec![m0]);
    }
    let mut shuffled = set.to_vec();
    shuffle_deterministic(&mut shuffled);
    tree_sumset_restricted(&shuffled, cfg)
}

/// `W(M, E)` via [`sumset_restricted`]; lower bound if truncated.
pub fn lookup_w_restricted(set: &[u64], m: usize, e_target: u64) -> Option<u128> {
    lookup_w_restricted_with_config(set, m, e_target, &LookupConfig::default())
}

/// Like [`lookup_w_restricted`], with explicit memory budget.
pub fn lookup_w_restricted_with_config(
    set: &[u64],
    m: usize,
    e_target: u64,
    cfg: &LookupConfig,
) -> Option<u128> {
    if m > set.len() {
        return Some(0);
    }
    Some(sumset_restricted(set, cfg).get(m, e_target))
}

/// log of [`lookup_w_restricted`].
pub fn log_lookup_w_restricted(set: &[u64], m: usize, e_target: u64) -> Option<f64> {
    Some(log_count(lookup_w_restricted(set, m, e_target)?))
}

/// Recursive divide-and-conquer for the restricted sumset.
fn tree_sumset_restricted(set: &[u64], cfg: &LookupConfig) -> RestrictedSumset {
    if let Some(dense) = try_direct_dense_restricted(set, cfg) {
        return dense;
    }
    if set.len() <= LEAF_SIZE {
        return leaf_sumset_restricted(set);
    }
    let mid = set.len() / 2;
    let left = tree_sumset_restricted(&set[..mid], cfg);
    let right = tree_sumset_restricted(&set[mid..], cfg);
    convolve_restricted(left, right, cfg)
}

/// Leaf: enumerate `2^k` subsets, group counts by `(size, sum)`.
fn leaf_sumset_restricted(set: &[u64]) -> RestrictedSumset {
    let n = set.len();
    let mut by_size: Vec<HashMap<u64, u128>> = vec![HashMap::new(); n + 1];
    for mask in 0..(1u64 << n) {
        let m = mask.count_ones() as usize;
        let sum: u64 = (0..n)
            .filter(|&j| mask & (1 << j) != 0)
            .map(|j| set[j])
            .sum();
        *by_size[m].entry(sum).or_insert(0) += 1;
    }
    RestrictedSumset::Complete(by_size)
}

/// Restricted variant of [`try_direct_dense`]; runs 2D `dp[m][e]` then materializes as HashMap slices.
fn try_direct_dense_restricted(set: &[u64], cfg: &LookupConfig) -> Option<RestrictedSumset> {
    let n = set.len();
    let sz = dense_admission_sz(set, cfg, n + 1)?;
    let mut dp = vec![vec![0u128; sz]; n + 1];
    dp[0][0] = 1;
    for &val in set {
        let v = val as usize;
        for mm in (1..=n).rev() {
            for j in (v..sz).rev() {
                dp[mm][j] += dp[mm - 1][j - v];
            }
        }
    }
    let by_size: Vec<HashMap<u64, u128>> = dp
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .filter(|&(_, c)| c > 0)
                .map(|(i, c)| (i as u64, c))
                .collect()
        })
        .collect();
    Some(RestrictedSumset::Complete(by_size))
}

// Convolve

/// Non-truncated sumset; type-level guarantee that `densify`/`into_map_complete` never see Truncated.
enum CompleteSumset {
    Sparse(HashMap<u64, u128>),
    Dense(DenseSumset),
}

impl CompleteSumset {
    fn len(&self) -> usize {
        match self {
            Self::Sparse(m) => m.len(),
            Self::Dense(d) => d.len(),
        }
    }

    fn max_sum(&self) -> Option<u64> {
        match self {
            Self::Sparse(m) => m.keys().copied().max(),
            Self::Dense(d) => d.max_sum(),
        }
    }
}

/// Density-aware convolution; `Truncated` propagates monotonically.
fn convolve(a: Sumset, b: Sumset, cfg: &LookupConfig) -> Sumset {
    let a = match into_complete(a) {
        Ok(complete) => complete,
        Err(am) => {
            return Sumset::Truncated(convolve_truncated_maps(
                am.into_inner(),
                into_map(b),
                cfg.max_entries,
            ));
        }
    };
    let b = match into_complete(b) {
        Ok(complete) => complete,
        Err(bm) => {
            return Sumset::Truncated(convolve_truncated_maps(
                into_map_complete(a),
                bm.into_inner(),
                cfg.max_entries,
            ));
        }
    };

    let projected_entries = a.len().saturating_mul(b.len());
    if projected_entries >= cfg.max_entries {
        return Sumset::Truncated(convolve_truncated_maps(
            into_map_complete(a),
            into_map_complete(b),
            cfg.max_entries,
        ));
    }

    let projected_max_sum = a
        .max_sum()
        .unwrap_or(0)
        .saturating_add(b.max_sum().unwrap_or(0));
    let density = density_estimate(projected_entries, projected_max_sum);
    if density > DENSE_THRESHOLD {
        let da = densify(a);
        let db = densify(b);
        match convolve_dense(&da, &db) {
            Some(dense) => Sumset::Dense(dense),
            None => Sumset::Truncated(convolve_truncated_maps(
                dense_to_map(da.offset, da.counts),
                dense_to_map(db.offset, db.counts),
                cfg.max_entries,
            )),
        }
    } else {
        let am = into_map_complete(a);
        let bm = into_map_complete(b);
        match convolve_sparse(&am, &bm) {
            Some(map) => Sumset::Sparse(map),
            None => Sumset::Truncated(convolve_truncated_maps(am, bm, cfg.max_entries)),
        }
    }
}

/// Convolves two sparse sumsets; `None` on `u64` sum or `u128` count overflow.
fn convolve_sparse(a: &HashMap<u64, u128>, b: &HashMap<u64, u128>) -> Option<HashMap<u64, u128>> {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut result: HashMap<u64, u128> = HashMap::with_capacity(small.len() * large.len());
    for (&s1, &c1) in small {
        for (&s2, &c2) in large {
            let sum = s1.checked_add(s2)?;
            let prod = c1.checked_mul(c2)?;
            let entry = result.entry(sum).or_insert(0);
            *entry = entry.checked_add(prod)?;
        }
    }
    Some(result)
}

/// Convolves two dense sumsets; `None` on offset, length, or `u128` overflow.
fn convolve_dense(a: &DenseSumset, b: &DenseSumset) -> Option<DenseSumset> {
    let offset = a.offset.checked_add(b.offset)?;
    let len = a
        .counts
        .len()
        .checked_add(b.counts.len())
        .and_then(|n| n.checked_sub(1))?;
    let mut counts = vec![0u128; len];
    let mut nonzero_count = 0usize;
    for (i, &ca) in a.counts.iter().enumerate() {
        if ca == 0 {
            continue;
        }
        for (j, &cb) in b.counts.iter().enumerate() {
            if cb == 0 {
                continue;
            }
            let prod = ca.checked_mul(cb)?;
            let was_zero = counts[i + j] == 0;
            counts[i + j] = counts[i + j].checked_add(prod)?;
            if was_zero {
                nonzero_count += 1;
            }
        }
    }
    Some(DenseSumset {
        offset,
        counts,
        nonzero_count,
    })
}

/// Truncated convolution; stops at `max_entries`, drops overflow contributions to keep the lower bound.
fn convolve_truncated_maps(
    a: HashMap<u64, u128>,
    b: HashMap<u64, u128>,
    max_entries: usize,
) -> LowerBound {
    let mut result: HashMap<u64, u128> = HashMap::new();
    for (&s1, &c1) in &a {
        for (&s2, &c2) in &b {
            let (Some(sum), Some(prod)) = (s1.checked_add(s2), c1.checked_mul(c2)) else {
                continue;
            };
            let entry = result.entry(sum).or_insert(0);
            if let Some(new) = entry.checked_add(prod) {
                *entry = new;
            }
            if result.len() >= max_entries {
                return LowerBound(result);
            }
        }
    }
    LowerBound(result)
}

/// Convolves restricted sumsets; `Truncated` propagates from operands, `max_entries`, or overflow.
fn convolve_restricted(
    a: RestrictedSumset,
    b: RestrictedSumset,
    cfg: &LookupConfig,
) -> RestrictedSumset {
    let a_truncated = matches!(a, RestrictedSumset::Truncated(_));
    let b_truncated = matches!(b, RestrictedSumset::Truncated(_));
    let a_slices = match a {
        RestrictedSumset::Complete(v) => v,
        RestrictedSumset::Truncated(t) => t.into_inner(),
    };
    let b_slices = match b {
        RestrictedSumset::Complete(v) => v,
        RestrictedSumset::Truncated(t) => t.into_inner(),
    };

    let result_len = a_slices.len() + b_slices.len() - 1;
    let mut accumulators: Vec<HashMap<u64, u128>> = vec![HashMap::new(); result_len];
    let mut hit_max_entries = false;
    let mut hit_overflow = false;

    'outer: for (m_a, slice_a) in a_slices.iter().enumerate() {
        for (m_b, slice_b) in b_slices.iter().enumerate() {
            let acc = &mut accumulators[m_a + m_b];
            for (&s_a, &c_a) in slice_a {
                for (&s_b, &c_b) in slice_b {
                    let (Some(sum), Some(prod)) = (s_a.checked_add(s_b), c_a.checked_mul(c_b))
                    else {
                        hit_overflow = true;
                        continue;
                    };
                    let entry = acc.entry(sum).or_insert(0);
                    if let Some(new) = entry.checked_add(prod) {
                        *entry = new;
                    } else {
                        hit_overflow = true;
                    }
                    if acc.len() >= cfg.max_entries {
                        hit_max_entries = true;
                        break 'outer;
                    }
                }
            }
        }
    }

    if a_truncated || b_truncated || hit_max_entries || hit_overflow {
        RestrictedSumset::Truncated(RestrictedLowerBound(accumulators))
    } else {
        RestrictedSumset::Complete(accumulators)
    }
}

/// Fraction of the sum range covered by entries; `0.0` when the range is empty.
fn density_estimate(entries: usize, max_sum: u64) -> f64 {
    if max_sum == 0 {
        return 0.0;
    }
    entries as f64 / (max_sum as f64 + 1.0)
}

/// Upper bound on sumset size: `min(2^n, sum + 1)` with `n` clamped to 64.
fn entries_upper_bound(n: usize, sum: u64) -> u128 {
    let subsets_limit = 1u128 << n.min(64);
    subsets_limit.min(sum as u128 + 1)
}

/// `Ok(CompleteSumset)` for non-truncated, `Err(LowerBound)` for `Truncated`.
fn into_complete(s: Sumset) -> Result<CompleteSumset, LowerBound> {
    match s {
        Sumset::Sparse(m) => Ok(CompleteSumset::Sparse(m)),
        Sumset::Dense(d) => Ok(CompleteSumset::Dense(d)),
        Sumset::Truncated(t) => Err(t),
    }
}

/// Materializes any `Sumset` as a `HashMap`; Dense is unpacked.
fn into_map(s: Sumset) -> HashMap<u64, u128> {
    match s {
        Sumset::Sparse(m) => m,
        Sumset::Truncated(t) => t.into_inner(),
        Sumset::Dense(d) => dense_to_map(d.offset, d.counts),
    }
}

fn into_map_complete(s: CompleteSumset) -> HashMap<u64, u128> {
    match s {
        CompleteSumset::Sparse(m) => m,
        CompleteSumset::Dense(d) => dense_to_map(d.offset, d.counts),
    }
}

fn dense_to_map(offset: u64, counts: Vec<u128>) -> HashMap<u64, u128> {
    counts
        .into_iter()
        .enumerate()
        .filter(|&(_, c)| c > 0)
        .map(|(i, c)| {
            let sum = offset
                .checked_add(i as u64)
                .expect("dense_to_map sum overflow");
            (sum, c)
        })
        .collect()
}

/// Materializes a `CompleteSumset` as a `DenseSumset`; caller must verify density > [`DENSE_THRESHOLD`].
fn densify(s: CompleteSumset) -> DenseSumset {
    match s {
        CompleteSumset::Dense(d) => d,
        CompleteSumset::Sparse(m) => {
            let min = m.keys().copied().min().unwrap_or(0);
            let max = m.keys().copied().max().unwrap_or(0);
            let len = usize::try_from(max - min)
                .ok()
                .and_then(|n| n.checked_add(1))
                .expect("densify: range exceeds usize");
            let mut counts = vec![0u128; len];
            let mut nonzero_count = 0;
            for (s, c) in m {
                counts[(s - min) as usize] = c;
                if c > 0 {
                    nonzero_count += 1;
                }
            }
            DenseSumset {
                offset: min,
                counts,
                nonzero_count,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    #[test]
    fn test_lookup_config_default_matches_const() {
        let cfg = LookupConfig::default();
        assert_eq!(cfg.max_entries, DEFAULT_MAX_ENTRIES);
        assert_eq!(cfg.memory_bytes(), DEFAULT_MAX_ENTRIES * ENTRY_SIZE_BYTES);
    }

    #[test]
    fn test_lookup_config_constructors_round_trip() {
        let from_mem = LookupConfig::from_memory_bytes(1_000_000);
        assert_eq!(from_mem.max_entries, 1_000_000 / ENTRY_SIZE_BYTES);

        let from_entries = LookupConfig::from_max_entries(1000);
        assert_eq!(from_entries.max_entries, 1000);
        assert_eq!(from_entries.memory_bytes(), 1000 * ENTRY_SIZE_BYTES);
    }

    #[test]
    fn test_empty_input_sumset_is_zero_count_one() {
        let s = sumset(&[], &LookupConfig::default());
        assert_eq!(s.get(0), 1);
        assert_eq!(s.get(1), 0);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_sumset_brute_force_match() {
        let a: Vec<u64> = (1..=12).collect();
        let s = sumset(&a, &LookupConfig::default());
        let total: u64 = a.iter().sum();
        for e in 0..=total {
            let exact = brute_force_w(&a, e).unwrap();
            assert_eq!(
                s.get(e),
                exact,
                "E={}: lookup={}, brute={}",
                e,
                s.get(e),
                exact
            );
        }
    }

    #[test]
    fn test_truncated_is_lower_bound() {
        let a: Vec<u64> = (1..=20).collect();
        let n = a.len();
        let tight = LookupConfig::from_max_entries(64);
        let s = sumset(&a, &tight);
        assert!(matches!(s, Sumset::Truncated(_)));

        let mut exact: HashMap<u64, u128> = HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *exact.entry(sum).or_insert(0) += 1;
        }
        for (&e, &w) in &exact {
            assert!(s.get(e) <= w, "E={}: bound={} > exact={}", e, s.get(e), w);
        }
    }

    #[test]
    fn test_sumset_max_sum_truncated_is_none() {
        let a: Vec<u64> = (1..=20).collect();
        let s = sumset(&a, &LookupConfig::from_max_entries(64));
        assert!(matches!(s, Sumset::Truncated(_)));
        assert_eq!(s.max_sum(), None);
    }

    #[test]
    fn test_dense_get_offset_correct() {
        let counts = vec![1u128, 0, 2, 3];
        let s = Sumset::dense(10, counts);
        assert_eq!(s.get(9), 0);
        assert_eq!(s.get(10), 1);
        assert_eq!(s.get(11), 0);
        assert_eq!(s.get(12), 2);
        assert_eq!(s.get(13), 3);
        assert_eq!(s.get(14), 0);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn test_log_lookup_w_zero_count_is_neg_inf() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        let lw = log_lookup_w(&a, 15).unwrap();
        assert!(lw.is_infinite() && lw.is_sign_negative());
    }

    #[test]
    fn test_log_lookup_w_finite_for_positive_count() {
        let a: Vec<u64> = vec![1, 2, 3, 4];
        let lw = log_lookup_w(&a, 5).unwrap();
        assert!(lw.is_finite() && lw >= 0.0);
    }

    #[test]
    fn test_sumset_short_circuits_dense_to_direct_dp() {
        // Small sum range → high density → direct dense DP, skipping tree.
        let a: Vec<u64> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let s = sumset(&a, &LookupConfig::default());
        assert!(matches!(s, Sumset::Dense { .. }));
        for k in 0..=a.len() as u64 {
            let expected: u128 = (0..=a.len() as u32)
                .filter(|&i| i == k as u32)
                .map(|i| {
                    (0..i).fold(1u128, |r, j| {
                        r * (a.len() as u128 - j as u128) / (j as u128 + 1)
                    })
                })
                .sum();
            assert_eq!(s.get(k), expected, "k={}", k);
        }
    }

    #[test]
    fn test_sumset_invariant_under_input_ordering() {
        // Shuffle determinism: same multiset, different order, same sumset.
        let a: Vec<u64> = vec![3, 7, 11, 5, 9, 2, 4, 6, 8, 10];
        let mut b = a.clone();
        b.reverse();
        let cfg = LookupConfig::default();
        let total: u64 = a.iter().sum();
        for e in [0, 1, 5, total / 2, total - 1, total] {
            assert_eq!(sumset(&a, &cfg).get(e), sumset(&b, &cfg).get(e), "E={}", e);
        }
    }

    #[test]
    fn test_shuffle_deterministic_reproducible() {
        let mut a: Vec<u64> = (1..=20).collect();
        let mut b: Vec<u64> = (1..=20).collect();
        shuffle_deterministic(&mut a);
        shuffle_deterministic(&mut b);
        assert_eq!(a, b, "same input must produce same shuffle");
    }

    #[test]
    fn test_brute_force_w() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9];
        let mut count: u128 = 0;
        for mask in 0..(1u64 << a.len()) {
            let sum: u64 = (0..a.len())
                .filter(|&j| mask & (1 << j) != 0)
                .map(|j| a[j])
                .sum();
            if sum == 15 {
                count += 1;
            }
        }
        assert_eq!(brute_force_w(&a, 15), Some(count));
        assert_eq!(brute_force_w(&a, 0), Some(1));
        assert_eq!(brute_force_w(&a, 35), Some(1));
        assert_eq!(brute_force_w(&a, 36), Some(0));
    }

    #[test]
    fn test_brute_force_w_too_large_returns_none() {
        let a: Vec<u64> = vec![1; 26];
        assert!(brute_force_w(&a, 5).is_none());
    }

    #[test]
    fn test_brute_force_w_restricted_too_large_returns_none() {
        let a: Vec<u64> = vec![1; 26];
        assert!(brute_force_w_restricted(&a, 5, 5).is_none());
    }

    #[test]
    fn test_dp_matches_brute_force() {
        // One 2^N enumeration into a HashMap, then check each E. Cheaper than
        // calling brute_force_w once per E.
        let a: Vec<u64> = (1..=20).collect();
        let n = a.len();

        let mut w_exact: HashMap<u64, u128> = HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry(sum).or_insert(0) += 1;
        }

        for (&e, &w) in &w_exact {
            let dp = dp_w(&a, e, 1_000_000).unwrap();
            assert_eq!(dp, w, "E={}: brute={}, dp={}", e, w, dp);
        }
    }

    #[test]
    fn test_dp_w_gcd_normalization() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        let dp = dp_w(&a, 30, 1_000_000).unwrap();
        let brute = brute_force_w(&a, 30).unwrap();
        assert_eq!(dp, brute);
    }

    #[test]
    fn test_dp_w_e_not_multiple_of_gcd_is_zero() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(dp_w(&a, 15, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_w_e_above_sum_is_zero() {
        let a: Vec<u64> = vec![1, 2, 3];
        let sum: u64 = a.iter().sum();
        assert_eq!(dp_w(&a, sum + 1, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_w_empty_input_is_none() {
        assert!(dp_w(&[], 0, 1_000_000).is_none());
    }

    #[test]
    fn test_dp_w_all_zero_is_none() {
        assert!(dp_w(&[0, 0, 0], 0, 1_000_000).is_none());
    }

    #[test]
    fn test_dp_w_sum_overflow_is_none() {
        let a: Vec<u64> = vec![u64::MAX, 1];
        assert!(dp_w(&a, 0, 1_000_000_000).is_none());
    }

    #[test]
    fn test_dp_w_max_cells_boundary() {
        let a: Vec<u64> = vec![1, 2, 3];
        let sum: u64 = a.iter().sum();
        let sz = sum as usize + 1;
        assert!(dp_w(&a, sum, sz).is_some());
        assert!(dp_w(&a, sum, sz - 1).is_none());
    }

    #[test]
    fn test_log_dp_w_zero_count_is_neg_inf() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        let lw = log_dp_w(&a, 15, 1_000_000).unwrap();
        assert!(lw.is_infinite() && lw.is_sign_negative());
    }

    #[test]
    fn test_log_dp_w_finite_for_positive_count() {
        let a: Vec<u64> = vec![1, 2, 3, 4];
        let lw = log_dp_w(&a, 5, 1_000_000).unwrap();
        assert!(lw.is_finite() && lw >= 0.0);
    }

    proptest! {
        #[test]
        fn proptest_dp_matches_brute_force(
            set in prop::collection::vec(1u64..=10, 1..=8),
            e in 0u64..=80,
        ) {
            let dp = dp_w(&set, e, 1_000_000).unwrap();
            let brute = brute_force_w(&set, e).unwrap();
            prop_assert_eq!(dp, brute);
        }

        #[test]
        fn proptest_dp_gcd_scale_invariant(
            set in prop::collection::vec(1u64..=10, 1..=8),
            c in 1u64..=10,
            e in 0u64..=80,
        ) {
            let scaled: Vec<u64> = set.iter().map(|&v| v * c).collect();
            let base = dp_w(&set, e, 1_000_000).unwrap();
            let scaled_w = dp_w(&scaled, e * c, 1_000_000).unwrap();
            prop_assert_eq!(base, scaled_w);
        }

        #[test]
        fn proptest_lookup_matches_brute_force(
            set in prop::collection::vec(1u64..=10, 1..=8),
            e in 0u64..=80,
        ) {
            let lookup = lookup_w(&set, e).unwrap();
            let brute = brute_force_w(&set, e).unwrap();
            prop_assert_eq!(lookup, brute);
        }

        #[test]
        fn proptest_lookup_gcd_scale_invariant(
            set in prop::collection::vec(1u64..=10, 1..=8),
            c in 1u64..=10,
            e in 0u64..=80,
        ) {
            let scaled: Vec<u64> = set.iter().map(|&v| v * c).collect();
            let base = lookup_w(&set, e).unwrap();
            let scaled_w = lookup_w(&scaled, e * c).unwrap();
            prop_assert_eq!(base, scaled_w);
        }

        /// Tight `max_entries` forces `Truncated`; result must remain a lower bound.
        #[test]
        fn proptest_lookup_truncated_is_lower_bound(
            set in prop::collection::vec(1u64..=10, 8..=12),
            e in 0u64..=120,
        ) {
            let cfg = LookupConfig::from_max_entries(32);
            let lookup = lookup_w_with_config(&set, e, &cfg).unwrap();
            let brute = brute_force_w(&set, e).unwrap();
            prop_assert!(lookup <= brute);
        }

        /// `vec![1; n]` saturates the sum range, forcing the dense path.
        #[test]
        fn proptest_lookup_dense_matches_brute(
            n in 8usize..=14,
            e in 0u64..=14,
        ) {
            let set = vec![1u64; n];
            let lookup = lookup_w(&set, e).unwrap();
            let brute = brute_force_w(&set, e).unwrap();
            prop_assert_eq!(lookup, brute);
        }

        /// `max_sum()` must equal the largest sum yielded by `iter()` for non-truncated sumsets.
        #[test]
        fn proptest_sumset_max_sum_matches_iter(
            set in prop::collection::vec(1u64..=10, 1..=8),
        ) {
            let s = sumset(&set, &LookupConfig::default());
            let iter_max = s.iter().map(|(sum, _)| sum).max();
            prop_assert_eq!(iter_max, s.max_sum());
        }
    }

    #[test]
    fn test_convolve_dense_overflow_falls_back_to_truncated() {
        // Counts at u128::MAX force `convolve_dense` to return None (product overflow);
        // `convolve` must fall back to a truncated map instead of panicking.
        let a = Sumset::dense(0, vec![u128::MAX, u128::MAX]);
        let b = Sumset::dense(0, vec![u128::MAX, u128::MAX]);
        let result = convolve(a, b, &LookupConfig::default());
        assert!(matches!(result, Sumset::Truncated(_)));
    }

    #[test]
    fn test_convolve_sparse_overflow_falls_back_to_truncated() {
        // Sparse path with u128::MAX counts: product overflows; `convolve` must
        // degrade to a truncated lower bound, not panic or wrap.
        let mut am: HashMap<u64, u128> = HashMap::new();
        am.insert(0, u128::MAX);
        am.insert(1, u128::MAX);
        let mut bm: HashMap<u64, u128> = HashMap::new();
        bm.insert(0, u128::MAX);
        bm.insert(1, u128::MAX);
        let a = Sumset::Sparse(am);
        let b = Sumset::Sparse(bm);
        let result = convolve(a, b, &LookupConfig::default());
        assert!(matches!(result, Sumset::Truncated(_)));
    }

    #[test]
    fn test_convolve_restricted_overflow_marks_truncated() {
        // Two restricted sumsets with u128::MAX counts: `convolve_restricted`
        // must mark the result Truncated (overflowing contributions dropped).
        let mut slice_a = HashMap::new();
        slice_a.insert(0u64, u128::MAX);
        let mut slice_b = HashMap::new();
        slice_b.insert(0u64, u128::MAX);
        let a = RestrictedSumset::Complete(vec![slice_a]);
        let b = RestrictedSumset::Complete(vec![slice_b]);
        let result = convolve_restricted(a, b, &LookupConfig::default());
        assert!(matches!(result, RestrictedSumset::Truncated(_)));
    }

    #[test]
    fn test_dp_restricted_matches_brute_force() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9, 2, 4];
        let sum: u64 = a.iter().sum();
        for m in 0..=a.len() {
            for e in 0..=sum {
                let brute = brute_force_w_restricted(&a, m, e).unwrap();
                let dp = dp_w_restricted(&a, m, e, 1_000_000).unwrap();
                assert_eq!(dp, brute, "m={}, e={}: brute={}, dp={}", m, e, brute, dp);
            }
        }
    }

    #[test]
    fn test_dp_restricted_sum_over_m_matches_dp_w() {
        // Σ_{m=0..=N} W(m, E) must equal W(E).
        let a: Vec<u64> = (1..=12).collect();
        let sum: u64 = a.iter().sum();
        for e in [0, 1, 10, sum / 2, sum - 1, sum] {
            let w_total = dp_w(&a, e, 1_000_000).unwrap();
            let w_sum: u128 = (0..=a.len())
                .map(|m| dp_w_restricted(&a, m, e, 1_000_000).unwrap())
                .sum();
            assert_eq!(
                w_sum, w_total,
                "e={}: Σ_m W(m,e)={}, W(e)={}",
                e, w_sum, w_total
            );
        }
    }

    #[test]
    fn test_dp_restricted_edges() {
        let a: Vec<u64> = vec![5, 10, 15];
        assert_eq!(dp_w_restricted(&a, 0, 0, 1_000_000), Some(1));
        assert_eq!(dp_w_restricted(&a, 0, 5, 1_000_000), Some(0));
        assert_eq!(dp_w_restricted(&a, 4, 10, 1_000_000), Some(0));
        assert_eq!(dp_w_restricted(&[], 0, 0, 1_000_000), Some(1));
        assert_eq!(dp_w_restricted(&[], 1, 0, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_restricted_gcd() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(
            dp_w_restricted(&a, 2, 30, 1_000_000),
            brute_force_w_restricted(&a, 2, 30)
        );
        assert_eq!(dp_w_restricted(&a, 2, 15, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_restricted_too_large() {
        let a: Vec<u64> = vec![1, 2, 3];
        assert!(dp_w_restricted(&a, 2, 3, 2).is_none());
    }

    #[test]
    fn test_dp_w_restricted_sum_overflow_is_none() {
        let a: Vec<u64> = vec![u64::MAX, 1];
        assert!(dp_w_restricted(&a, 1, 0, 1_000_000_000).is_none());
    }

    #[test]
    fn test_log_dp_w_restricted_zero_count_is_neg_inf() {
        let a: Vec<u64> = vec![5, 10, 15];
        let lw = log_dp_w_restricted(&a, 2, 3, 1_000_000).unwrap();
        assert!(lw.is_infinite() && lw.is_sign_negative());
    }

    #[test]
    fn test_log_dp_w_restricted_finite_for_positive_count() {
        let a: Vec<u64> = vec![1, 2, 3, 4];
        let lw = log_dp_w_restricted(&a, 2, 5, 1_000_000).unwrap();
        assert!(lw.is_finite() && lw >= 0.0);
    }

    #[test]
    fn test_lookup_restricted_brute_force_match() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9, 2, 4];
        let sum: u64 = a.iter().sum();
        let s = sumset_restricted(&a, &LookupConfig::default());
        for m in 0..=a.len() {
            for e in 0..=sum {
                let brute = brute_force_w_restricted(&a, m, e).unwrap();
                assert_eq!(s.get(m, e), brute, "m={}, e={}", m, e);
            }
        }
    }

    #[test]
    fn test_lookup_restricted_sum_over_m_matches_unrestricted() {
        // Σ_{m=0..=N} W(m, E) must equal W(E).
        let a: Vec<u64> = (1..=12).collect();
        let sum: u64 = a.iter().sum();
        let cfg = LookupConfig::default();
        let restricted = sumset_restricted(&a, &cfg);
        let unrestricted = sumset(&a, &cfg);
        for e in [0, 1, 10, sum / 2, sum - 1, sum] {
            let w_sum: u128 = (0..=a.len()).map(|m| restricted.get(m, e)).sum();
            assert_eq!(w_sum, unrestricted.get(e), "e={}", e);
        }
    }

    #[test]
    fn test_sumset_iter_total_count_is_2_pow_n() {
        // Σ counts == 2^N; iter visits every reachable sum.
        let a: Vec<u64> = (1..=10).collect();
        let s = sumset(&a, &LookupConfig::default());
        let total: u128 = s.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 1u128 << a.len());
    }

    #[test]
    fn test_restricted_sumset_iter_total_count_is_2_pow_n() {
        let a: Vec<u64> = (1..=10).collect();
        let s = sumset_restricted(&a, &LookupConfig::default());
        let total: u128 = s.iter().map(|(_, _, c)| c).sum();
        assert_eq!(total, 1u128 << a.len());
        assert_eq!(s.num_sizes(), a.len() + 1);
    }

    #[test]
    fn test_sumset_restricted_short_circuits_dense() {
        // Saturated sum range forces `try_direct_dense_restricted`; Σ counts == 2^N
        // confirms the dense DP produced the full sumset.
        let a: Vec<u64> = vec![1; 16];
        let s = sumset_restricted(&a, &LookupConfig::default());
        let total: u128 = s.iter().map(|(_, _, c)| c).sum();
        assert_eq!(total, 1u128 << a.len());
    }

    #[test]
    fn test_restricted_sumset_max_sum_truncated_is_none() {
        let a: Vec<u64> = (1..=20).collect();
        let s = sumset_restricted(&a, &LookupConfig::from_max_entries(64));
        assert!(matches!(s, RestrictedSumset::Truncated(_)));
        assert_eq!(s.max_sum(), None);
    }

    proptest! {
        #[test]
        fn proptest_dp_restricted_matches_brute_force(
            set in prop::collection::vec(1u64..=10, 1..=6),
            m in 0usize..=6,
            e in 0u64..=60,
        ) {
            let dp = dp_w_restricted(&set, m, e, 1_000_000).unwrap();
            let brute = brute_force_w_restricted(&set, m, e).unwrap();
            prop_assert_eq!(dp, brute);
        }

        #[test]
        fn proptest_dp_restricted_gcd_scale_invariant(
            set in prop::collection::vec(1u64..=10, 1..=6),
            m in 0usize..=6,
            c in 1u64..=10,
            e in 0u64..=60,
        ) {
            let scaled: Vec<u64> = set.iter().map(|&v| v * c).collect();
            let base = dp_w_restricted(&set, m, e, 1_000_000).unwrap();
            let scaled_w = dp_w_restricted(&scaled, m, e * c, 1_000_000).unwrap();
            prop_assert_eq!(base, scaled_w);
        }

        #[test]
        fn proptest_lookup_restricted_matches_brute_force(
            set in prop::collection::vec(1u64..=10, 1..=6),
            m in 0usize..=6,
            e in 0u64..=60,
        ) {
            let lookup = lookup_w_restricted(&set, m, e).unwrap();
            let brute = brute_force_w_restricted(&set, m, e).unwrap();
            prop_assert_eq!(lookup, brute);
        }

        #[test]
        fn proptest_lookup_restricted_gcd_scale_invariant(
            set in prop::collection::vec(1u64..=10, 1..=6),
            m in 0usize..=6,
            c in 1u64..=10,
            e in 0u64..=60,
        ) {
            let scaled: Vec<u64> = set.iter().map(|&v| v * c).collect();
            let base = lookup_w_restricted(&set, m, e).unwrap();
            let scaled_w = lookup_w_restricted(&scaled, m, e * c).unwrap();
            prop_assert_eq!(base, scaled_w);
        }

        /// Same multiset, different orders, same restricted sumset entry.
        #[test]
        fn proptest_restricted_sumset_invariant_under_ordering(
            mut set in prop::collection::vec(1u64..=10, 1..=8),
            m in 0usize..=8,
            e in 0u64..=80,
        ) {
            let cfg = LookupConfig::default();
            let a = sumset_restricted(&set, &cfg).get(m, e);
            set.reverse();
            let b = sumset_restricted(&set, &cfg).get(m, e);
            prop_assert_eq!(a, b);
        }

        /// Tight `max_entries` forces `Truncated`; result must remain a lower bound.
        #[test]
        fn proptest_lookup_restricted_truncated_is_lower_bound(
            set in prop::collection::vec(1u64..=10, 8..=12),
            m in 0usize..=12,
            e in 0u64..=120,
        ) {
            let cfg = LookupConfig::from_max_entries(32);
            let lookup = lookup_w_restricted_with_config(&set, m, e, &cfg).unwrap();
            let brute = brute_force_w_restricted(&set, m, e).unwrap();
            prop_assert!(lookup <= brute);
        }
    }
}
