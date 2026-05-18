//! Sparse nonnegative convolution: BFN Algorithms 1, 3, 4, 5, 6 ([arXiv:2107.07625](https://arxiv.org/pdf/2107.07625)).
//! Alg 2 (CRT) and 7 (m=t⁶) not implemented: Goldilocks has 6 orders of margin past MAX_MONEY,
//! so Las Vegas with a fixed seed is already deterministic at CoinJoin scale.

mod cyclic_fold;
mod engine;
mod field;
mod hash;
mod ntt;
mod result;
mod schedules;

pub use field::Field;
pub use ntt::Goldilocks;
pub use result::{Convolution, Termination};

use engine::LasVegasRun;
use hash::LinearHash;
use schedules::alg6_schedule;
#[cfg(test)]
use schedules::{alg1_schedule, alg4_schedule};
use std::collections::HashMap;

/// Algorithm 3 (Lemma 16): bucketize + 1-sparsity test.
#[must_use]
pub(crate) fn bucketed_recover<F: Field>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    hash: &LinearHash,
) -> Vec<(u64, u64)> {
    bucketed_recover_in::<F>(a, b, hash)
}

/// Algorithm 5 (Lemma 20): A⋆B − C with prime hash.
pub(crate) fn bucketed_recover_residual<F: Field, I: IntoIterator<Item = (u64, u64)>>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    c: I,
    p: u64,
) -> Vec<(u64, u64)> {
    bucketed_recover_residual_in::<F, I>(a, b, c, p)
}

/// Algorithm 1 (Theorem 2): Las Vegas O(t · log²t). Cross-validation against Lemma 22 production path.
#[cfg(test)]
pub(crate) fn convolve_seeded<F: Field>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    seed: u64,
    max_inner_iters: u32,
) -> Convolution {
    sparse_convolve_seeded_in::<F>(a, b, seed, max_inner_iters)
}

/// Algorithm 4 (Lemma 19): ε-tail variant. Cross-validation only.
#[cfg(test)]
pub(crate) fn convolve_seeded_eps<F: Field>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    seed: u64,
    epsilon: f64,
    max_inner_iters: u32,
) -> Convolution {
    sparse_convolve_seeded_eps_in::<F>(a, b, seed, epsilon, max_inner_iters)
}

/// Algorithm 6 (Lemma 22): linear+prime hybrid, default for `GradedSumset`.
pub(crate) fn convolve_seeded_hybrid<F: Field>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    seed: u64,
    max_inner_iters: u32,
) -> Convolution {
    sparse_convolve_seeded_hybrid_in::<F>(a, b, seed, max_inner_iters)
}

/// `m = 2^23`: 6× u64 buffers at ~64 MB each.
const MAX_LOG_M: u32 = 23;

struct BucketTriple {
    values: Vec<u64>,
    derivative: Vec<u64>,
    second: Vec<u64>,
}

fn bucketize_field<F, I, H>(input: I, m: usize, hash_fn: H) -> BucketTriple
where
    F: Field,
    I: IntoIterator<Item = (u64, u64)>,
    H: Fn(u64) -> usize,
{
    let p = u128::from(F::P);
    let mut values = vec![0u128; m];
    let mut derivative = vec![0u128; m];
    let mut second = vec![0u128; m];
    for (x, c) in input {
        let bucket = hash_fn(x);
        let xp = u128::from(x % F::P);
        let cp = u128::from(c % F::P);
        let xc = xp * cp % p;
        let xxc = xp * xc % p;
        values[bucket] = (values[bucket] + cp) % p;
        derivative[bucket] = (derivative[bucket] + xc) % p;
        second[bucket] = (second[bucket] + xxc) % p;
    }
    let to_u64 = |v: Vec<u128>| {
        v.into_iter()
            .map(|x| u64::try_from(x).expect("entries are reduced mod F::P, fit u64"))
            .collect()
    };
    BucketTriple {
        values: to_u64(values),
        derivative: to_u64(derivative),
        second: to_u64(second),
    }
}

/// (x, y, z) triples for 1-sparsity test y² ≟ xz (Lemma 16).
fn recover_triple<F: Field>(
    bucket_a: &BucketTriple,
    bucket_b: &BucketTriple,
    bucket_count: usize,
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let x = F::cyclic_convolve(&bucket_a.values, &bucket_b.values, bucket_count);
    let y = sum2::<F>(
        &F::cyclic_convolve(&bucket_a.derivative, &bucket_b.values, bucket_count),
        &F::cyclic_convolve(&bucket_a.values, &bucket_b.derivative, bucket_count),
    );
    let z = sum3_double_middle::<F>(
        &F::cyclic_convolve(&bucket_a.second, &bucket_b.values, bucket_count),
        &F::cyclic_convolve(&bucket_a.derivative, &bucket_b.derivative, bucket_count),
        &F::cyclic_convolve(&bucket_a.values, &bucket_b.second, bucket_count),
    );
    (x, y, z)
}

fn sum2<F: Field>(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b).map(|(&x, &y)| F::add(x, y)).collect()
}

fn sum3_double_middle<F: Field>(a: &[u64], b: &[u64], c: &[u64]) -> Vec<u64> {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((&x, &y), &z)| F::add(F::add(F::add(x, y), y), z))
        .collect()
}

fn sub_vec<F: Field>(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b).map(|(&x, &y)| F::sub(x, y)).collect()
}

pub(crate) struct SplitMix(u64);

impl SplitMix {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn sum_counts(v: &[(u64, u64)]) -> u64 {
    v.iter().map(|&(_, c)| c).sum()
}

fn max_index(v: &[(u64, u64)]) -> Option<u64> {
    v.iter().map(|&(idx, _)| idx).max()
}

fn merge_max(acc: &mut HashMap<u64, u64>, contributions: Vec<(u64, u64)>, n_max: u64) {
    for (idx, c) in contributions {
        // idx > n_max means 1-sparsity false positive.
        if idx > n_max {
            continue;
        }
        let entry = acc.entry(idx).or_insert(0);
        *entry = (*entry).max(c);
    }
}

/// Algorithm 1 (Theorem 2): O(t log² t). `u32::MAX` for unbounded budget.
#[cfg(test)]
fn sparse_convolve_seeded_in<F: Field>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    seed: u64,
    max_inner_iters: u32,
) -> Convolution {
    let target = u128::from(sum_counts(a)) * u128::from(sum_counts(b));
    if target == 0 {
        return Convolution::empty_complete();
    }
    if target >= u128::from(F::P) {
        return Convolution::precondition_violated();
    }
    LasVegasRun::<F>::new(a, b, target, seed, max_inner_iters).execute(alg1_schedule())
}

/// Algorithm 4 (Lemma 19): O(t log² t) with (1−δ) tail bound.
#[cfg(test)]
fn sparse_convolve_seeded_eps_in<F: Field>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    seed: u64,
    epsilon: f64,
    max_inner_iters: u32,
) -> Convolution {
    assert!(epsilon > 0.0, "epsilon must be positive");
    let target = u128::from(sum_counts(a)) * u128::from(sum_counts(b));
    if target == 0 {
        return Convolution::empty_complete();
    }
    if target >= u128::from(F::P) {
        return Convolution::precondition_violated();
    }
    LasVegasRun::<F>::new(a, b, target, seed, max_inner_iters).execute(alg4_schedule(epsilon))
}

/// Algorithm 5 (Lemma 20): A⋆B − C with prime hash; caller ensures ≥ 0 pointwise.
fn bucketed_recover_residual_in<F: Field, I: IntoIterator<Item = (u64, u64)>>(
    a_in: &[(u64, u64)],
    b_in: &[(u64, u64)],
    c_in: I,
    prime: u64,
) -> Vec<(u64, u64)> {
    let bucket_count =
        usize::try_from(prime).expect("prime ≤ 2·m_prime ≤ 2^MAX_LOG_M+1 fits usize");
    let prime_hash = |x: u64| usize::try_from(x % prime).expect("x % prime < prime, fits usize");
    let ba = bucketize_field::<F, _, _>(a_in.iter().copied(), bucket_count, prime_hash);
    let bb = bucketize_field::<F, _, _>(b_in.iter().copied(), bucket_count, prime_hash);
    let bc = bucketize_field::<F, _, _>(c_in, bucket_count, prime_hash);

    let x_ab = cyclic_fold::cyclic_convolve::<F>(&ba.values, &bb.values, bucket_count);
    let x = sub_vec::<F>(&x_ab, &bc.values);

    let y_dab = cyclic_fold::cyclic_convolve::<F>(&ba.derivative, &bb.values, bucket_count);
    let y_adb = cyclic_fold::cyclic_convolve::<F>(&ba.values, &bb.derivative, bucket_count);
    let y = sub_vec::<F>(&sum2::<F>(&y_dab, &y_adb), &bc.derivative);

    let z_d2ab = cyclic_fold::cyclic_convolve::<F>(&ba.second, &bb.values, bucket_count);
    let z_dadb = cyclic_fold::cyclic_convolve::<F>(&ba.derivative, &bb.derivative, bucket_count);
    let z_ad2b = cyclic_fold::cyclic_convolve::<F>(&ba.values, &bb.second, bucket_count);
    let z = sub_vec::<F>(
        &sum3_double_middle::<F>(&z_d2ab, &z_dadb, &z_ad2b),
        &bc.second,
    );

    let mut recovered: HashMap<u64, u64> = HashMap::new();
    for k in 0..bucket_count {
        if x[k] == 0 {
            continue;
        }
        if F::mul(y[k], y[k]) != F::mul(x[k], z[k]) {
            continue;
        }
        let z_idx = F::mul(y[k], F::inv(x[k]));
        let entry = recovered.entry(z_idx).or_insert(0);
        *entry = entry.saturating_add(x[k]);
    }
    recovered.into_iter().collect()
}

/// Algorithm 3 (Lemma 16): bucketize + 1-sparsity test.
fn bucketed_recover_in<F: Field>(
    a_in: &[(u64, u64)],
    b_in: &[(u64, u64)],
    hash: &LinearHash,
) -> Vec<(u64, u64)> {
    let bucket_count = hash.m();
    let ba = bucketize_field::<F, _, _>(a_in.iter().copied(), bucket_count, |x| hash.eval(x));
    let bb = bucketize_field::<F, _, _>(b_in.iter().copied(), bucket_count, |x| hash.eval(x));
    let (x, y, z) = recover_triple::<F>(&ba, &bb, bucket_count);

    let mut recovered: HashMap<u64, u64> = HashMap::new();
    for k in 0..bucket_count {
        if x[k] == 0 {
            continue;
        }
        if F::mul(y[k], y[k]) != F::mul(x[k], z[k]) {
            continue;
        }
        let z_idx = F::mul(y[k], F::inv(x[k]));
        let entry = recovered.entry(z_idx).or_insert(0);
        *entry = entry.saturating_add(x[k]);
    }
    recovered.into_iter().collect()
}

/// Algorithm 6 (Lemma 22): linear+prime hybrid, O(t log t · log log n).
fn sparse_convolve_seeded_hybrid_in<F: Field>(
    a: &[(u64, u64)],
    b: &[(u64, u64)],
    seed: u64,
    max_inner_iters: u32,
) -> Convolution {
    let target = u128::from(sum_counts(a)) * u128::from(sum_counts(b));
    if target == 0 {
        return Convolution::empty_complete();
    }
    if target >= u128::from(F::P) {
        return Convolution::precondition_violated();
    }
    let n_max = max_index(a)
        .unwrap_or(0)
        .saturating_add(max_index(b).unwrap_or(0));
    let log_n = log2_ceil(n_max.saturating_add(2)).max(1) as usize;
    let log_log_n = log2_ceil(log_n as u64 + 1).max(1) as usize;
    let phase1_iters = 3 * log_log_n as u64;

    LasVegasRun::<F>::new(a, b, target, seed, max_inner_iters)
        .execute(alg6_schedule(log_n, phase1_iters))
}

fn log2_ceil(n: u64) -> u32 {
    if n <= 1 {
        return 0;
    }
    64 - (n - 1).leading_zeros()
}

pub(crate) fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut i = 5u64;
    while i.saturating_mul(i) <= n {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

pub(crate) fn sample_prime_in(low: u64, high: u64, rng: &mut SplitMix) -> u64 {
    debug_assert!(low <= high && low >= 2);
    let span = high - low + 1;
    loop {
        let candidate = low + rng.next() % span;
        if is_prime(candidate) {
            return candidate;
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
