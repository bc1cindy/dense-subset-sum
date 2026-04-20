//! Block-convolution and DP lower bounds on W(E). All estimators are sound.

use std::collections::HashMap;

use crate::sasamoto::{gcd_slice, log_w_signed_sasamoto};

/// Default sumset cap (~100MB at nominal entry size). Override via `LookupConfig`
/// (memory/entries) and/or `sat_per_bin` to trade exactness for range.
pub const DEFAULT_MAX_ENTRIES: usize = 4_194_304;

/// Nominal per-entry size: `u64` sum + `u128` count. Real `HashMap` has bucket
/// overhead on top, so this is a lower bound used to convert between the two
/// knobs; pick a memory budget conservatively.
const ENTRY_SIZE_BYTES: usize = std::mem::size_of::<(u64, u128)>();

/// Maximum sumset during block convolution. Both `max_memory_bytes` and
/// `max_entries` are exposed so callers can reason about whichever resource is
/// scarce; use `from_memory_bytes` or `from_max_entries` to keep them consistent.
/// `Default` reproduces 2^22-entry max (~100MB nominal).
///
/// `sat_per_bin` quantizes the sumset output into a histogram: each key is
/// `⌊sum / sat_per_bin⌋` and the stored count is the number of subsets whose
/// sum lands in that bucket. `sat_per_bin = 1` is exact point-W; `> 1`
/// chops the lower-order bits of the sumset, shrinking memory roughly
/// linearly. Inputs stay exact; binning happens on sumset keys after each
/// block join, so bin-aligned inputs incur no loss while unaligned inputs
/// carry a per-block-boundary rounding error of at most one bin.
#[derive(Debug, Clone)]
pub struct LookupConfig {
    pub max_memory_bytes: usize,
    pub max_entries: usize,
    pub sat_per_bin: u64,
}

impl Default for LookupConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: DEFAULT_MAX_ENTRIES * ENTRY_SIZE_BYTES,
            max_entries: DEFAULT_MAX_ENTRIES,
            sat_per_bin: 1,
        }
    }
}

impl LookupConfig {
    pub fn from_memory_bytes(bytes: usize) -> Self {
        Self {
            max_memory_bytes: bytes,
            max_entries: bytes / ENTRY_SIZE_BYTES,
            sat_per_bin: 1,
        }
    }

    pub fn from_max_entries(n: usize) -> Self {
        Self {
            max_memory_bytes: n * ENTRY_SIZE_BYTES,
            max_entries: n,
            sat_per_bin: 1,
        }
    }

    /// Quantize the output sumset into `bin`-sized buckets (`key = ⌊sum/bin⌋`).
    /// `1` disables binning; `≥ 2` returns histogram bucket counts instead of
    /// exact per-E W values.
    pub fn with_sat_per_bin(mut self, bin: u64) -> Self {
        self.sat_per_bin = bin.max(1);
        self
    }
}

/// Exact W(E) by enumerating 2^N subsets. Max N ≈ 25 in practice.
pub fn brute_force_w(a: &[u64], e_target: u64) -> u64 {
    let n = a.len();
    assert!(n <= 30, "brute_force_w: N={} too large (max 30)", n);
    let mut count = 0u64;
    for mask in 0..(1u64 << n) {
        let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
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
pub fn lookup_w(a: &[u64], e_target: u64, block_size: usize) -> Option<u128> {
    lookup_w_with_config(a, e_target, block_size, &LookupConfig::default())
}

/// Like `lookup_w`, but with an explicit memory/entry max.
///
/// When `cfg.sat_per_bin > 1`, the sumset keys are quantized to
/// `⌊sum / sat_per_bin⌋` after each block join and the returned count is the
/// histogram bucket at `⌊e_target / sat_per_bin⌋`, i.e. the number of
/// subsets whose sum lies in `[k·bin, (k+1)·bin)`, not point W(E).
pub fn lookup_w_with_config(
    a: &[u64],
    e_target: u64,
    block_size: usize,
    cfg: &LookupConfig,
) -> Option<u128> {
    if a.is_empty() {
        return None;
    }
    let bin = cfg.sat_per_bin.max(1);
    let combined = full_sumset(a, block_size, cfg.max_entries, bin);
    Some(*combined.get(&(e_target / bin)).unwrap_or(&0))
}

pub fn log_lookup_w(a: &[u64], e_target: u64, block_size: usize) -> Option<f64> {
    log_lookup_w_with_config(a, e_target, block_size, &LookupConfig::default())
}

/// Like `log_lookup_w`, but with an explicit memory/entry max.
pub fn log_lookup_w_with_config(
    a: &[u64],
    e_target: u64,
    block_size: usize,
    cfg: &LookupConfig,
) -> Option<f64> {
    let w = lookup_w_with_config(a, e_target, block_size, cfg)?;
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

/// Like `log_lookup_w_signed_target_aware`, but with an explicit memory/entry max.
///
/// When `cfg.sat_per_bin > 1`, each side's sumset is quantized into
/// `⌊sum / sat_per_bin⌋` histogram buckets and `target` is mapped to
/// `⌊target / sat_per_bin⌋` (sign-preserving). The result is the log of the
/// bucket count at the signed bin offset, not point W.
pub fn log_lookup_w_signed_target_aware_with_config(
    positives: &[u64],
    negatives: &[u64],
    target: i64,
    block_size: usize,
    cfg: &LookupConfig,
) -> Option<f64> {
    let bin = cfg.sat_per_bin.max(1);
    let target_q: i64 = target / bin as i64;

    let abs_target = target_q.unsigned_abs();
    let pos_target = if target_q >= 0 { abs_target } else { 0 };
    let neg_target = if target_q < 0 { abs_target } else { 0 };

    let pos_sums = full_sumset_near_target(positives, block_size, cfg.max_entries, pos_target, bin);
    let neg_sums = full_sumset_near_target(negatives, block_size, cfg.max_entries, neg_target, bin);

    let mut total: u128 = 0;
    for (&s_pos, &c_pos) in &pos_sums {
        let s_neg_required = (s_pos as i64).checked_sub(target_q)?;
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

/// Sasamoto for N ≥ 40, target-aware lookup otherwise. Recommended signed entry point.
pub fn log_w_signed_adaptive(
    positives: &[u64],
    negatives: &[u64],
    target: i64,
    block_size: usize,
) -> Option<f64> {
    log_w_signed_adaptive_with_config(
        positives,
        negatives,
        target,
        block_size,
        &LookupConfig::default(),
    )
}

/// Like `log_w_signed_adaptive`, but with an explicit memory/entry cap.
pub fn log_w_signed_adaptive_with_config(
    positives: &[u64],
    negatives: &[u64],
    target: i64,
    block_size: usize,
    cfg: &LookupConfig,
) -> Option<f64> {
    let n_total = positives.len() + negatives.len();
    if n_total >= 40
        && let Some(v) = log_w_signed_sasamoto(positives, negatives, target)
        && v.is_finite()
    {
        return Some(v);
    }
    log_lookup_w_signed_target_aware_with_config(positives, negatives, target, block_size, cfg)
        .filter(|v| v.is_finite())
}

/// Exact W(E) via DP: O(N · sum_max). `None` when sum_max exceeds `max_table_size`.
///
/// Returns `u128` to represent the exact count up to 2^N. Casting to `f64`
/// loses precision above 2^53; fine for error ratios, not for equality.
pub fn dp_w(a: &[u64], e_target: u64, max_table_size: usize) -> Option<u128> {
    if a.is_empty() {
        return None;
    }

    let g = gcd_slice(a);
    if g == 0 {
        return None;
    }
    if !e_target.is_multiple_of(g) {
        return Some(0);
    }

    let a_norm: Vec<u64> = a.iter().map(|&v| v / g).collect();
    let e_norm = e_target / g;
    let sum_max: u64 = a_norm.iter().sum();

    if e_norm > sum_max {
        return Some(0);
    }
    if sum_max as usize > max_table_size {
        return None;
    }

    let sz = sum_max as usize + 1;
    let mut dp = vec![0u128; sz];
    dp[0] = 1;

    for &val in &a_norm {
        let v = val as usize;
        for j in (v..sz).rev() {
            dp[j] += dp[j - v];
        }
    }

    Some(dp[e_norm as usize])
}

pub fn log_dp_w(a: &[u64], e_target: u64, max_table_size: usize) -> Option<f64> {
    let w = dp_w(a, e_target, max_table_size)?;
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

/// Quantize sumset keys into `⌊k / bin⌋` buckets, merging counts. Identity when `bin ≤ 1`.
fn bin_keys(m: HashMap<u64, u128>, bin: u64) -> HashMap<u64, u128> {
    if bin <= 1 {
        return m;
    }
    let mut result: HashMap<u64, u128> = HashMap::with_capacity(m.len());
    for (k, c) in m {
        *result.entry(k / bin).or_insert(0) += c;
    }
    result
}

/// Block-convolved sumset with output-level binning. Each block's exact sumset
/// is quantized to `⌊sum / bin⌋` before joining the accumulator, so the
/// accumulator is in bin-index space throughout. `bin = 1` is exact point-W;
/// `bin > 1` yields a histogram with up to ±1 bin rounding per block join.
/// Over `max_entries`, remaining elements fold naively in bin space.
/// Callers must pass `bin >= 1`; the public wrappers clamp before calling.
fn full_sumset(a: &[u64], block_size: usize, max_entries: usize, bin: u64) -> HashMap<u64, u128> {
    if a.is_empty() {
        let mut m = HashMap::new();
        m.insert(0u64, 1u128);
        return m;
    }
    let k = block_size.max(1).min(a.len());
    let blocks: Vec<&[u64]> = a.chunks(k).collect();
    let mut combined = bin_keys(block_sumset(blocks[0]), bin);
    for block in &blocks[1..] {
        let block_sums = if combined.len() >= max_entries {
            let mut coarse = HashMap::new();
            coarse.insert(0u64, 1u128);
            for &v in *block {
                let v_bin = v / bin;
                let mut next = coarse.clone();
                for (&s, &c) in &coarse {
                    *next.entry(s + v_bin).or_insert(0) += c;
                }
                coarse = next;
                if coarse.len() >= max_entries {
                    break;
                }
            }
            coarse
        } else {
            bin_keys(block_sumset(block), bin)
        };
        combined = convolve(&combined, &block_sums, max_entries);
        if combined.len() >= max_entries {
            break;
        }
    }
    combined
}

/// Proximity-pruned variant: keeps entries closest to `target` (expressed in
/// bin-index space) when over `max_entries`. Binning semantics match
/// `full_sumset`; callers must pass `bin >= 1`.
fn full_sumset_near_target(
    a: &[u64],
    block_size: usize,
    max_entries: usize,
    target: u64,
    bin: u64,
) -> HashMap<u64, u128> {
    if a.is_empty() {
        let mut m = HashMap::new();
        m.insert(0u64, 1u128);
        return m;
    }
    let k = block_size.max(1).min(a.len());
    let blocks: Vec<&[u64]> = a.chunks(k).collect();
    let mut combined = bin_keys(block_sumset(blocks[0]), bin);
    for block in &blocks[1..] {
        let block_sums = if combined.len() >= max_entries {
            let mut coarse = HashMap::new();
            coarse.insert(0u64, 1u128);
            for &v in *block {
                let v_bin = v / bin;
                let mut next = coarse.clone();
                for (&s, &c) in &coarse {
                    *next.entry(s + v_bin).or_insert(0) += c;
                }
                coarse = next;
                if coarse.len() >= max_entries {
                    break;
                }
            }
            coarse
        } else {
            bin_keys(block_sumset(block), bin)
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
    fn test_lookup_config_default_sat_per_bin_is_one() {
        assert_eq!(LookupConfig::default().sat_per_bin, 1);
    }

    #[test]
    fn test_with_sat_per_bin_sets_field_and_clamps_zero() {
        assert_eq!(LookupConfig::default().with_sat_per_bin(8).sat_per_bin, 8);
        assert_eq!(
            LookupConfig::default().with_sat_per_bin(0).sat_per_bin,
            1,
            "bin=0 should clamp to 1"
        );
    }

    #[test]
    fn test_sat_per_bin_one_is_identity() {
        let a: Vec<u64> = (1..=16).collect();
        let e = a.iter().sum::<u64>() / 2;
        let base = lookup_w(&a, e, 4).unwrap();
        let binned =
            lookup_w_with_config(&a, e, 4, &LookupConfig::default().with_sat_per_bin(1)).unwrap();
        assert_eq!(base, binned, "sat_per_bin=1 must not alter output");
    }

    #[test]
    fn test_sat_per_bin_lossless_on_bin_aligned_inputs() {
        let a: Vec<u64> = (1..=12).map(|i| i * 8).collect();
        let e = a.iter().sum::<u64>() / 2;
        let exact = brute_force_w(&a, e) as u128;
        let cfg = LookupConfig::default().with_sat_per_bin(8);
        let binned = lookup_w_with_config(&a, e, 4, &cfg).unwrap();
        assert_eq!(
            binned, exact,
            "bin-aligned inputs must yield exact W under binning"
        );
    }

    #[test]
    fn test_sat_per_bin_recovers_exact_under_tight_cap_when_aligned() {
        // Large sat-scale values where the unbinned sumset would bail at a
        // tight cap. With bin = common divisor, sums collide into ≤ 20 * 20 + 1
        // quantized entries — well under the cap — so lookup recovers exact W.
        let a: Vec<u64> = (1..=20).map(|i| i * 10_000).collect();
        let e = a.iter().sum::<u64>() / 2;
        let exact = brute_force_w(&a, e) as u128;

        let binned = LookupConfig::from_max_entries(4096).with_sat_per_bin(10_000);
        let w_binned = lookup_w_with_config(&a, e, 4, &binned).unwrap();
        assert_eq!(
            w_binned, exact,
            "bin-aligned values with bin = coin size should recover exact W"
        );
    }

    #[test]
    fn test_sat_per_bin_output_histogram_unaligned() {
        // a = [3, 5, 7] has exact sumset {0, 3, 5, 7, 8, 10, 12, 15}, all count 1.
        // With bin = 5, the histogram groups sums into bins [k·5, (k+1)·5):
        //   bucket 0 → {0, 3}      = 2 subsets
        //   bucket 1 → {5, 7, 8}   = 3 subsets
        //   bucket 2 → {10, 12}    = 2 subsets
        //   bucket 3 → {15}        = 1 subset
        // With block_size = N the full sumset is computed exactly before binning,
        // so the bucket counts are exact even though inputs aren't bin-aligned.
        let a: Vec<u64> = vec![3, 5, 7];
        let cfg = LookupConfig::default().with_sat_per_bin(5);
        let w0 = lookup_w_with_config(&a, 0, a.len(), &cfg).unwrap();
        let w5 = lookup_w_with_config(&a, 5, a.len(), &cfg).unwrap();
        let w10 = lookup_w_with_config(&a, 10, a.len(), &cfg).unwrap();
        let w15 = lookup_w_with_config(&a, 15, a.len(), &cfg).unwrap();
        assert_eq!(
            (w0, w5, w10, w15),
            (2, 3, 2, 1),
            "block_size = N must yield exact bucket counts under output binning"
        );

        // Sanity: bucket counts sum to the full subset count 2^3 = 8.
        assert_eq!(w0 + w5 + w10 + w15, 8);
    }

    #[test]
    fn test_signed_sat_per_bin_matches_unscaled_when_aligned() {
        let pos: Vec<u64> = vec![100, 200, 300];
        let neg: Vec<u64> = vec![100, 200];
        let target: i64 = 100;

        let cfg = LookupConfig::default().with_sat_per_bin(100);
        let lw_binned =
            log_lookup_w_signed_target_aware_with_config(&pos, &neg, target, 3, &cfg).unwrap();

        let lw_unscaled =
            log_lookup_w_signed_target_aware(&[1, 2, 3], &[1, 2], 1, 3, 4_194_304).unwrap();
        assert!(
            (lw_binned - lw_unscaled).abs() < 1e-9,
            "bin-aligned signed lookup must match the unscaled problem: binned={} unscaled={}",
            lw_binned,
            lw_unscaled
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

    #[test]
    fn test_log_w_signed_adaptive_picks_best() {
        let pos: Vec<u64> = vec![10, 20, 30];
        let neg: Vec<u64> = vec![10, 20, 30];
        let adaptive = log_w_signed_adaptive(&pos, &neg, 0, 3);
        assert!(adaptive.is_some());
        let pos_big: Vec<u64> = (1..=50).map(|i| i * 100).collect();
        let neg_big: Vec<u64> = (1..=50).map(|i| i * 100).collect();
        let adaptive_big = log_w_signed_adaptive(&pos_big, &neg_big, 0, 10);
        assert!(adaptive_big.is_some());
        assert!(adaptive_big.unwrap() > 0.0);
    }
}
