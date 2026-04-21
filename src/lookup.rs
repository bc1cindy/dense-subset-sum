//! Sumset counts W(E) = #{S ⊆ A : ΣS = E}, three redundant ways:
//!
//! - [`brute_force_w`]: exact, N ≲ 25.
//! - [`dp_w`]: exact DP, bails past `max_table_size`.
//! - [`lookup_w`]: block-convolution lower bound, scales via [`LookupConfig`].
//!
//! Signed variants ([`log_w_signed`]) handle ΣS − ΣT = target. Tests
//! cross-check all three; the `estimator` module falls back lookup ← dp when
//! DP overflows.

use std::collections::HashMap;

use crate::sasamoto::{gcd_slice, log_w_signed_sasamoto};

/// Explicit method choice for signed probes. Pure routing — no combination logic.
/// The caller picks based on deployment (WASM/mobile/server) and protocol stage
/// (precomputation vs. critical section). Callers that want to compare both
/// methods invoke the primitives (`log_w_signed_sasamoto`,
/// `log_lookup_w_signed_target_aware`) directly and compose themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedMethod {
    /// Asymptotic estimate (O(N)). Fast; unreliable for small N or low density.
    Sasamoto,
    /// Target-aware lookup (O(2^N / blocks)). Sound lower bound; can hit the
    /// `DEFAULT_MAX_ENTRIES` cap on large N with sat-scale values.
    Lookup,
}

/// Routes a signed probe to one of the two primitives based on `method`. No
/// auto-selection, no combination: see [`SignedMethod`]. Uses
/// `LookupConfig::default()`; see [`log_w_signed_with_config`] for an explicit cap.
pub fn log_w_signed(
    positives: &[u64],
    negatives: &[u64],
    target: i64,
    block_size: usize,
    method: SignedMethod,
) -> Option<f64> {
    log_w_signed_with_config(
        positives,
        negatives,
        target,
        block_size,
        &LookupConfig::default(),
        method,
    )
}

/// Like [`log_w_signed`], but with an explicit memory/entry cap for the lookup path.
/// The cap is ignored when `method == SignedMethod::Sasamoto`.
pub fn log_w_signed_with_config(
    positives: &[u64],
    negatives: &[u64],
    target: i64,
    block_size: usize,
    cfg: &LookupConfig,
    method: SignedMethod,
) -> Option<f64> {
    match method {
        SignedMethod::Sasamoto => {
            log_w_signed_sasamoto(positives, negatives, target).filter(|v| v.is_finite())
        }
        SignedMethod::Lookup => log_lookup_w_signed_target_aware_with_config(
            positives, negatives, target, block_size, cfg,
        )
        .filter(|v| v.is_finite()),
    }
}

/// 2^22 ≈ 4M entries fits in ~100MB; unbounded N≥20 with sat-scale values hits 10^9 entries (OOM).
pub const DEFAULT_MAX_ENTRIES: usize = 4_194_304;

/// Nominal per-entry size: `u64` sum + `u128` count. Real `HashMap` has bucket
/// overhead on top, so this is a lower bound used to convert between the two
/// knobs; pick a memory budget conservatively.
const ENTRY_SIZE_BYTES: usize = std::mem::size_of::<(u64, u128)>();

/// Maximum sumset during block convolution. Both `max_memory_bytes` and
/// `max_entries` are exposed so callers can reason about whichever resource is
/// scarce; use `from_memory_bytes` or `from_max_entries` to keep them consistent.
/// `Default` reproduces 2^22-entry max (~100MB nominal).
#[derive(Debug, Clone)]
pub struct LookupConfig {
    pub max_memory_bytes: usize,
    pub max_entries: usize,
}

impl Default for LookupConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: DEFAULT_MAX_ENTRIES * ENTRY_SIZE_BYTES,
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }
}

impl LookupConfig {
    pub fn from_memory_bytes(bytes: usize) -> Self {
        Self {
            max_memory_bytes: bytes,
            max_entries: bytes / ENTRY_SIZE_BYTES,
        }
    }

    pub fn from_max_entries(n: usize) -> Self {
        Self {
            max_memory_bytes: n * ENTRY_SIZE_BYTES,
            max_entries: n,
        }
    }
}

/// Exact W(E) by enumerating 2^N subsets. Max N ≈ 25 in practice.
pub fn brute_force_w(original_set: &[u64], e_target: u64) -> u64 {
    let n = original_set.len();
    assert!(n <= 30, "brute_force_w: N={} too large (max 30)", n);
    let mut count = 0u64;
    for mask in 0..(1u64 << n) {
        let sum: u64 = (0..n)
            .filter(|&j| mask & (1 << j) != 0)
            .map(|j| original_set[j])
            .sum();
        if sum == e_target {
            count += 1;
        }
    }
    count
}

/// Block-convolution lower bound on W(E). k=N: exact; k=1: loose; k=15..20: practical sweet spot.
///
/// Returns `u128` because `W > 2^64` is reachable for N ≥ 64. Callers that cast
/// to `f64` lose precision above 2^53 (f64 mantissa); that's acceptable for
/// ratios/error reports but not for equality checks on large W.
pub fn lookup_w(original_set: &[u64], e_target: u64, block_size: usize) -> Option<u128> {
    lookup_w_with_config(original_set, e_target, block_size, &LookupConfig::default())
}

/// Like `lookup_w`, but with an explicit memory/entry max.
pub fn lookup_w_with_config(
    original_set: &[u64],
    e_target: u64,
    block_size: usize,
    cfg: &LookupConfig,
) -> Option<u128> {
    if original_set.is_empty() {
        return None;
    }
    let combined = full_sumset(original_set, block_size, cfg.max_entries);
    Some(*combined.get(&e_target).unwrap_or(&0))
}

pub fn log_lookup_w(original_set: &[u64], e_target: u64, block_size: usize) -> Option<f64> {
    log_lookup_w_with_config(original_set, e_target, block_size, &LookupConfig::default())
}

/// Distinct reachable sums |{ΣS : S ⊆ A}|, bounded by `cfg.max_entries`.
/// Conservative under the cap: the bound is a sub-sumset, so the true size
/// may be larger when this returns `cfg.max_entries`.
pub fn sumset_size_with_config(
    original_set: &[u64],
    block_size: usize,
    cfg: &LookupConfig,
) -> usize {
    full_sumset(original_set, block_size, cfg.max_entries).len()
}

/// Like `log_lookup_w`, but with an explicit memory/entry max.
pub fn log_lookup_w_with_config(
    original_set: &[u64],
    e_target: u64,
    block_size: usize,
    cfg: &LookupConfig,
) -> Option<f64> {
    let w = lookup_w_with_config(original_set, e_target, block_size, cfg)?;
    if w == 0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some((w as f64).ln())
    }
}

/// Signed subset-sum lower bound: log #pairs (S ⊆ positives, T ⊆ negatives) with ΣS − ΣT = target.
/// Proximity-pruned near `target`. CoinJoin convention: positives=outputs, negatives=inputs.
pub fn log_lookup_w_signed_target_aware(
    positives: &[u64],
    negatives: &[u64],
    target: i64,
    block_size: usize,
    max_entries: usize,
) -> Option<f64> {
    log_lookup_w_signed_target_aware_with_config(
        positives,
        negatives,
        target,
        block_size,
        &LookupConfig::from_max_entries(max_entries),
    )
}

/// Like `log_lookup_w_signed_target_aware`, but with an explicit memory/entry cap.
pub fn log_lookup_w_signed_target_aware_with_config(
    positives: &[u64],
    negatives: &[u64],
    target: i64,
    block_size: usize,
    cfg: &LookupConfig,
) -> Option<f64> {
    let abs_target = target.unsigned_abs();
    let pos_target = if target >= 0 { abs_target } else { 0 };
    let neg_target = if target < 0 { abs_target } else { 0 };

    let pos_sums = full_sumset_near_target(positives, block_size, cfg.max_entries, pos_target);
    let neg_sums = full_sumset_near_target(negatives, block_size, cfg.max_entries, neg_target);

    let mut total: u128 = 0;
    for (&s_pos, &c_pos) in &pos_sums {
        let s_neg_required = (s_pos as i64).checked_sub(target)?;
        if s_neg_required < 0 {
            continue;
        }
        if let Some(&c_neg) = neg_sums.get(&(s_neg_required as u64)) {
            total = total.saturating_add(c_pos.saturating_mul(c_neg));
        }
    }
    if total == 0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some((total as f64).ln())
    }
}

/// Exact W(E) via DP: O(N · sum_max). `None` when sum_max exceeds `max_table_size`.
///
/// `original_set` is the input multiset (values to choose subsets of), not a
/// precomputed sumset. Exact alternative to [`lookup_w`], kept for cross-validation —
/// the two methods converge on the same count via different paths.
///
/// Returns `u128` to represent the exact count up to 2^N. Casting to `f64`
/// loses precision above 2^53; fine for error ratios, not for equality.
pub fn dp_w(original_set: &[u64], e_target: u64, max_table_size: usize) -> Option<u128> {
    if original_set.is_empty() {
        return None;
    }

    let g = gcd_slice(original_set);
    if g == 0 {
        return None;
    }
    if !e_target.is_multiple_of(g) {
        return Some(0);
    }

    let normalized: Vec<u64> = original_set.iter().map(|&v| v / g).collect();
    let e_norm = e_target / g;
    let sum_max: u64 = normalized.iter().sum();

    if e_norm > sum_max {
        return Some(0);
    }
    if sum_max as usize > max_table_size {
        return None;
    }

    let sz = sum_max as usize + 1;
    let mut dp = vec![0u128; sz];
    dp[0] = 1;

    for &val in &normalized {
        let v = val as usize;
        for j in (v..sz).rev() {
            dp[j] += dp[j - v];
        }
    }

    Some(dp[e_norm as usize])
}

pub fn log_dp_w(original_set: &[u64], e_target: u64, max_table_size: usize) -> Option<f64> {
    let w = dp_w(original_set, e_target, max_table_size)?;
    if w == 0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some((w as f64).ln())
    }
}

/// Exact W(M, E): #subsets of size exactly `m` summing to `e_target`.
///
/// 2D DP: O(N · M · sum_max) time, O(M · sum_max) space. `None` when the table
/// would exceed `max_table_size` cells. Identity: `Σ_{m=0..=N} W(m,E) = W(E)`.
/// Used by the per-input and cluster penalty terms where subset size is bounded.
pub fn dp_w_restricted(
    original_set: &[u64],
    m: usize,
    e_target: u64,
    max_table_size: usize,
) -> Option<u128> {
    if m > original_set.len() {
        return Some(0);
    }
    if m == 0 {
        return Some(if e_target == 0 { 1 } else { 0 });
    }

    let g = gcd_slice(original_set);
    if g == 0 {
        return None;
    }
    if !e_target.is_multiple_of(g) {
        return Some(0);
    }

    let normalized: Vec<u64> = original_set.iter().map(|&v| v / g).collect();
    let e_norm = e_target / g;
    let sum_max: u64 = normalized.iter().sum();

    if e_norm > sum_max {
        return Some(0);
    }

    let sz = sum_max as usize + 1;
    let cells = (m + 1).checked_mul(sz)?;
    if cells > max_table_size {
        return None;
    }

    let mut dp = vec![vec![0u128; sz]; m + 1];
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
    max_table_size: usize,
) -> Option<f64> {
    let w = dp_w_restricted(original_set, m, e_target, max_table_size)?;
    if w == 0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some((w as f64).ln())
    }
}

/// Exact sumset of a small block: HashMap<sum, count> over all 2^n subsets.
fn block_sumset(block: &[u64]) -> HashMap<u64, u128> {
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

/// Minkowski convolution; bails at `max_entries` — result is a sub-sumset, still a lower bound on W.
fn convolve_capped(
    a: &HashMap<u64, u128>,
    b: &HashMap<u64, u128>,
    max_entries: usize,
) -> HashMap<u64, u128> {
    let mut result: HashMap<u64, u128> = HashMap::new();
    for (&s1, &c1) in a {
        for (&s2, &c2) in b {
            *result.entry(s1 + s2).or_insert(0) += c1 * c2;
        }
        if result.len() >= max_entries {
            break;
        }
    }
    result
}

/// When over `max_entries`, keeps entries closest to `target` — tighter lower bound on W(target).
fn convolve_capped_near_target(
    a: &HashMap<u64, u128>,
    b: &HashMap<u64, u128>,
    max_entries: usize,
    target: u64,
) -> HashMap<u64, u128> {
    let mut result: HashMap<u64, u128> = HashMap::new();
    for (&s1, &c1) in a {
        for (&s2, &c2) in b {
            *result.entry(s1 + s2).or_insert(0) += c1 * c2;
        }
    }
    if result.len() <= max_entries {
        return result;
    }
    let mut entries: Vec<(u64, u128)> = result.into_iter().collect();
    entries.sort_by_key(|&(s, _)| (s as i64 - target as i64).unsigned_abs());
    entries.truncate(max_entries);
    entries.into_iter().collect()
}

fn convolve(
    a: &HashMap<u64, u128>,
    b: &HashMap<u64, u128>,
    max_entries: usize,
) -> HashMap<u64, u128> {
    convolve_capped(a, b, max_entries)
}

/// Block-convolved sumset. Over `max_entries`, remaining blocks fold naively as {0, v_i} — still a valid W lower bound.
fn full_sumset(original_set: &[u64], block_size: usize, max_entries: usize) -> HashMap<u64, u128> {
    if original_set.is_empty() {
        let mut m = HashMap::new();
        m.insert(0u64, 1u128);
        return m;
    }
    let k = block_size.max(1).min(original_set.len());
    let blocks: Vec<&[u64]> = original_set.chunks(k).collect();
    let mut combined = block_sumset(blocks[0]);
    for block in &blocks[1..] {
        let block_sums = if combined.len() >= max_entries {
            let mut coarse = HashMap::new();
            coarse.insert(0u64, 1u128);
            for &v in *block {
                let mut next = coarse.clone();
                for (&s, &c) in &coarse {
                    *next.entry(s + v).or_insert(0) += c;
                }
                coarse = next;
                if coarse.len() >= max_entries {
                    break;
                }
            }
            coarse
        } else {
            block_sumset(block)
        };
        combined = convolve(&combined, &block_sums, max_entries);
        if combined.len() >= max_entries {
            break;
        }
    }
    combined
}

/// Proximity-pruned variant: keeps entries closest to `target` when over `max_entries`.
fn full_sumset_near_target(
    original_set: &[u64],
    block_size: usize,
    max_entries: usize,
    target: u64,
) -> HashMap<u64, u128> {
    if original_set.is_empty() {
        let mut m = HashMap::new();
        m.insert(0u64, 1u128);
        return m;
    }
    let k = block_size.max(1).min(original_set.len());
    let blocks: Vec<&[u64]> = original_set.chunks(k).collect();
    let mut combined = block_sumset(blocks[0]);
    for block in &blocks[1..] {
        let block_sums = if combined.len() >= max_entries {
            let mut coarse = HashMap::new();
            coarse.insert(0u64, 1u128);
            for &v in *block {
                let mut next = coarse.clone();
                for (&s, &c) in &coarse {
                    *next.entry(s + v).or_insert(0) += c;
                }
                coarse = next;
                if coarse.len() >= max_entries {
                    break;
                }
            }
            coarse
        } else {
            block_sumset(block)
        };
        combined = convolve_capped_near_target(&combined, &block_sums, max_entries, target);
    }
    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_config_default_matches_historical_max() {
        let cfg = LookupConfig::default();
        assert_eq!(cfg.max_entries, DEFAULT_MAX_ENTRIES);
        assert_eq!(cfg.max_memory_bytes, DEFAULT_MAX_ENTRIES * ENTRY_SIZE_BYTES);
    }

    #[test]
    fn test_lookup_config_constructors_round_trip() {
        let from_mem = LookupConfig::from_memory_bytes(1_000_000);
        assert_eq!(from_mem.max_memory_bytes, 1_000_000);
        assert_eq!(from_mem.max_entries, 1_000_000 / ENTRY_SIZE_BYTES);

        let from_entries = LookupConfig::from_max_entries(1000);
        assert_eq!(from_entries.max_entries, 1000);
        assert_eq!(from_entries.max_memory_bytes, 1000 * ENTRY_SIZE_BYTES);
    }

    #[test]
    fn test_lookup_config_smaller_cap_is_looser_lower_bound() {
        let a: Vec<u64> = (1..=20).collect();
        let e = a.iter().sum::<u64>() / 2;

        let tight = LookupConfig::from_max_entries(64);
        let w_tight = lookup_w_with_config(&a, e, 4, &tight).unwrap();
        let w_default = lookup_w_with_config(&a, e, 4, &LookupConfig::default()).unwrap();
        let exact = brute_force_w(&a, e) as u128;

        assert!(
            w_tight <= w_default,
            "tighter cap must produce ≤ W: tight={}, default={}",
            w_tight,
            w_default
        );
        assert!(
            w_default <= exact,
            "lookup must remain a lower bound: default={}, exact={}",
            w_default,
            exact
        );
    }

    #[test]
    fn test_convolve_capped_near_target_retains_target() {
        let mut a = HashMap::new();
        for i in 0..100u64 {
            a.insert(i, 1u128);
        }
        let mut b = HashMap::new();
        b.insert(0, 1u128);
        b.insert(50, 1u128);
        let result = convolve_capped_near_target(&a, &b, 50, 75);
        assert!(result.len() <= 50);
        assert!(
            result.contains_key(&75),
            "target=75 should be retained in capped result"
        );
    }

    #[test]
    fn test_brute_force_w_u64() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9];
        let mut count = 0u64;
        for mask in 0..(1u64 << a.len()) {
            let sum: u64 = (0..a.len())
                .filter(|&j| mask & (1 << j) != 0)
                .map(|j| a[j])
                .sum();
            if sum == 15 {
                count += 1;
            }
        }
        assert_eq!(brute_force_w(&a, 15), count);
        assert_eq!(brute_force_w(&a, 0), 1);
        assert_eq!(brute_force_w(&a, 35), 1);
        assert_eq!(brute_force_w(&a, 36), 0);
    }

    #[test]
    fn test_lookup_exact_when_k_equals_n() {
        let a: Vec<u64> = (1..=16).collect();
        let n = a.len();

        for e in 1..a.iter().sum() {
            let exact = brute_force_w(&a, e);
            let lookup = lookup_w(&a, e, n).unwrap();
            assert_eq!(
                lookup, exact as u128,
                "E={}: brute={}, lookup={}",
                e, exact, lookup
            );
        }
    }

    #[test]
    fn test_lookup_is_lower_bound() {
        let a: Vec<u64> = (1..=20).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let exact = brute_force_w(&a, e_mid);

        for k in &[1, 3, 5, 10, 15, 20] {
            let lb = lookup_w(&a, e_mid, *k).unwrap();
            assert!(
                lb <= exact as u128,
                "k={}: lookup={} > exact={}",
                k,
                lb,
                exact
            );
        }
    }

    #[test]
    fn test_lookup_monotone_in_k() {
        let a: Vec<u64> = (1..=16).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;

        let mut prev = 0u128;
        for k in 1..=a.len() {
            let w = lookup_w(&a, e_mid, k).unwrap();
            assert!(w >= prev, "k={}: {} < prev {}", k, w, prev);
            prev = w;
        }
    }

    #[test]
    fn test_dp_matches_brute_force() {
        let a: Vec<u64> = (1..=20).collect();
        let n = a.len();

        let mut w_exact: HashMap<u64, u64> = HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry(sum).or_insert(0) += 1;
        }

        for (&e, &w) in &w_exact {
            let dp = dp_w(&a, e, 1_000_000).unwrap();
            assert_eq!(dp, w as u128, "E={}: brute={}, dp={}", e, w, dp);
        }
    }

    #[test]
    fn test_dp_gcd() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(
            dp_w(&a, 30, 1_000_000).unwrap(),
            brute_force_w(&a, 30) as u128
        );
        assert_eq!(dp_w(&a, 15, 1_000_000).unwrap(), 0);
    }

    #[test]
    fn test_dp_too_large() {
        let a: Vec<u64> = vec![1, 2];
        assert!(dp_w(&a, 1, 2).is_none());
    }

    fn brute_force_w_restricted(a: &[u64], m: usize, e: u64) -> u128 {
        let n = a.len();
        let mut count = 0u128;
        for mask in 0..(1u64 << n) {
            if mask.count_ones() as usize != m {
                continue;
            }
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            if sum == e {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn test_dp_restricted_matches_brute_force() {
        let a: Vec<u64> = vec![3, 7, 11, 5, 9, 2, 4];
        let sum: u64 = a.iter().sum();
        for m in 0..=a.len() {
            for e in 0..=sum {
                let brute = brute_force_w_restricted(&a, m, e);
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
            assert_eq!(w_sum, w_total, "e={}: Σ_m W(m,e)={}, W(e)={}", e, w_sum, w_total);
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
        // a = 10·[1,2,3,4]. W(2, 30) over scaled = W(2, 3) over base = C(3-choose-pairs summing to 3).
        let a: Vec<u64> = vec![10, 20, 30, 40];
        assert_eq!(
            dp_w_restricted(&a, 2, 30, 1_000_000),
            Some(brute_force_w_restricted(&a, 2, 30))
        );
        assert_eq!(dp_w_restricted(&a, 2, 15, 1_000_000), Some(0));
    }

    #[test]
    fn test_dp_restricted_too_large() {
        let a: Vec<u64> = vec![1, 2, 3];
        assert!(dp_w_restricted(&a, 2, 3, 2).is_none());
    }

    #[test]
    fn test_log_dp_restricted_zero_count_is_neg_inf() {
        let a: Vec<u64> = vec![5, 10, 15];
        let lw = log_dp_w_restricted(&a, 2, 3, 1_000_000).unwrap();
        assert!(lw.is_infinite() && lw.is_sign_negative());
    }

    fn brute_signed(pos: &[u64], neg: &[u64], target: i64) -> u128 {
        let mut total = 0u128;
        let np = pos.len();
        let nn = neg.len();
        for sp in 0u32..(1 << np) {
            let ss: i64 = (0..np)
                .filter(|i| (sp >> i) & 1 == 1)
                .map(|i| pos[i] as i64)
                .sum();
            for sn in 0u32..(1 << nn) {
                let sn_sum: i64 = (0..nn)
                    .filter(|j| (sn >> j) & 1 == 1)
                    .map(|j| neg[j] as i64)
                    .sum();
                if ss - sn_sum == target {
                    total += 1;
                }
            }
        }
        total
    }

    #[test]
    fn test_lookup_w_signed_vs_brute() {
        let pos: Vec<u64> = vec![1, 2, 3];
        let neg: Vec<u64> = vec![1, 2];
        for target in -3..=6 {
            let brute = brute_signed(&pos, &neg, target);
            let lw =
                log_lookup_w_signed_target_aware(&pos, &neg, target, pos.len(), 1_000_000).unwrap();
            let lookup: u128 = if lw.is_finite() {
                lw.exp().round() as u128
            } else {
                0
            };
            assert_eq!(
                brute, lookup,
                "target={}: brute={} lookup={}",
                target, brute, lookup
            );
        }
    }

    #[test]
    fn test_lookup_w_signed_target_zero_has_empty_pair() {
        let pos: Vec<u64> = vec![5, 7];
        let neg: Vec<u64> = vec![3, 11];
        let lw = log_lookup_w_signed_target_aware(&pos, &neg, 0, pos.len(), 1_000_000).unwrap();
        assert!(
            lw.is_finite(),
            "expected at least the empty pair, got {}",
            lw
        );
    }

    #[test]
    fn test_log_lookup_w_signed_zero_count_is_neg_inf() {
        let pos: Vec<u64> = vec![1, 2];
        let neg: Vec<u64> = vec![4];
        let lw = log_lookup_w_signed_target_aware(&pos, &neg, -100, 2, 1_000_000).unwrap();
        assert!(lw.is_infinite() && lw.is_sign_negative(), "got {}", lw);
    }

    #[test]
    fn test_lookup_w_signed_target_aware_nonneg() {
        let pos: Vec<u64> = vec![10, 20, 30];
        let neg: Vec<u64> = vec![10, 20, 30];
        let lw = log_lookup_w_signed_target_aware(&pos, &neg, 0, 3, 4_194_304).unwrap();
        assert!(lw.is_finite(), "balanced multisets should have W >= 1");
    }
}
