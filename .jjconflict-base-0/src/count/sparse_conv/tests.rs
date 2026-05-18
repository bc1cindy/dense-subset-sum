use super::*;
use proptest::prelude::*;
use std::collections::HashMap;

fn brute_force(a: &[(u64, u64)], b: &[(u64, u64)]) -> HashMap<u64, u64> {
    let mut out = HashMap::new();
    for &(x, ax) in a {
        for &(y, by) in b {
            *out.entry(x + y).or_insert(0u64) += ax * by;
        }
    }
    out
}

fn r_to_map(r: &[(u64, u64)]) -> HashMap<u64, u64> {
    let mut out = HashMap::new();
    for &(idx, c) in r {
        *out.entry(idx).or_insert(0u64) += c;
    }
    out
}

fn lower_bound_holds(r: &[(u64, u64)], truth: &HashMap<u64, u64>) -> bool {
    r_to_map(r)
        .iter()
        .all(|(idx, &c)| c <= *truth.get(idx).unwrap_or(&0))
}

#[test]
fn empty_inputs() {
    let h = LinearHash::new(0xdead_beef, 4);
    assert!(bucketed_recover::<Goldilocks>(&[], &[], &h).is_empty());
}

#[test]
fn single_pair_recovers_at_correct_index() {
    let a = vec![(2u64, 1u64)];
    let b = vec![(3u64, 1u64)];
    let h = LinearHash::new(0xdead_beef, 4);
    let r = bucketed_recover::<Goldilocks>(&a, &b, &h);
    for (idx, c) in &r {
        assert_eq!(*idx, 5, "1-sparse buckets must recover 2 + 3");
        assert_eq!(*c, 1);
    }
    assert!(!r.is_empty());
}

#[test]
fn weighted_single_pair() {
    let a = vec![(4u64, 5u64)];
    let b = vec![(11u64, 7u64)];
    let h = LinearHash::new(0xfeed_face, 4);
    let r = bucketed_recover::<Goldilocks>(&a, &b, &h);
    for (idx, c) in &r {
        assert_eq!(*idx, 15);
        assert_eq!(*c, 35);
    }
    assert!(!r.is_empty());
}

#[test]
fn r_is_lower_bound_across_seeds() {
    let a: Vec<(u64, u64)> = (0..10).map(|i| (i * 7 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..10).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);

    for seed in 0u64..32 {
        let h = LinearHash::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), 64);
        let r = bucketed_recover::<Goldilocks>(&a, &b, &h);
        assert!(lower_bound_holds(&r, &truth), "seed={seed}");
    }
}

#[test]
fn coordwise_max_recovers_most_truth_when_m_exceeds_t() {
    let a: Vec<(u64, u64)> = (0..10).map(|i| (i * 7 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..10).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);
    let m = 256;

    let mut acc: HashMap<u64, u64> = HashMap::new();
    for seed in 0u64..16 {
        let h = LinearHash::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1, m);
        for (idx, c) in bucketed_recover::<Goldilocks>(&a, &b, &h) {
            let entry = acc.entry(idx).or_insert(0);
            *entry = (*entry).max(c);
        }
    }
    let recovered = truth
        .iter()
        .filter(|(idx, c)| acc.get(*idx).copied().unwrap_or(0) == **c)
        .count();
    let frac = recovered as f64 / truth.len() as f64;
    assert!(
        frac > 0.9,
        "recovered {recovered}/{} ({frac:.2})",
        truth.len()
    );
}

#[test]
fn weighted_inputs_match_truth() {
    let a = vec![(2u64, 3u64), (5, 1), (11, 2)];
    let b = vec![(7u64, 4u64), (13, 5)];
    let truth = brute_force(&a, &b);
    let m = 64;

    let mut acc: HashMap<u64, u64> = HashMap::new();
    for seed in 0u64..32 {
        let h = LinearHash::new(seed.wrapping_mul(0xdead_beef_cafe_babe) | 1, m);
        for (idx, c) in bucketed_recover::<Goldilocks>(&a, &b, &h) {
            let entry = acc.entry(idx).or_insert(0);
            *entry = (*entry).max(c);
        }
    }
    for (idx, &c) in &truth {
        assert_eq!(acc.get(idx).copied().unwrap_or(0), c, "idx={idx}");
    }
}

fn assert_matches_truth(actual: &[(u64, u64)], truth: &HashMap<u64, u64>) {
    let actual_map: HashMap<u64, u64> = actual.iter().copied().collect();
    assert_eq!(actual_map.len(), truth.len(), "support size differs");
    for (idx, &c) in truth {
        assert_eq!(actual_map.get(idx).copied(), Some(c), "idx={idx}");
    }
}

fn run_until_complete(a: &[(u64, u64)], b: &[(u64, u64)], seed: u64) -> Convolution {
    let out = convolve_seeded::<Goldilocks>(a, b, seed, u32::MAX);
    assert!(
        out.termination() == Termination::Complete,
        "expected exact termination"
    );
    out
}

#[test]
fn sparse_convolve_empty_returns_empty() {
    let out = convolve_seeded::<Goldilocks>(&[], &[(1, 1)], 0, u32::MAX);
    assert!(out.termination() == Termination::Complete && out.support.is_empty());
    let out = convolve_seeded::<Goldilocks>(&[(1, 1)], &[], 0, u32::MAX);
    assert!(out.termination() == Termination::Complete && out.support.is_empty());
    let out = convolve_seeded_eps::<Goldilocks>(&[], &[(1, 1)], 0, 1.0, u32::MAX);
    assert!(out.termination() == Termination::Complete && out.support.is_empty());
}

#[test]
fn sparse_convolve_single_pair() {
    let out = run_until_complete(&[(2, 1)], &[(3, 1)], 42);
    assert_eq!(out.support, vec![(5, 1)]);
}

#[test]
fn sparse_convolve_matches_truth_boolean() {
    let a: Vec<(u64, u64)> = (0..8).map(|i| (i * 3 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..6).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);
    for seed in 0u64..4 {
        let out = run_until_complete(&a, &b, seed);
        assert_matches_truth(&out.support, &truth);
    }
}

#[test]
fn sparse_convolve_matches_truth_weighted() {
    let a = vec![(2u64, 3u64), (5, 1), (11, 2), (17, 4)];
    let b = vec![(7u64, 4u64), (13, 5), (19, 1)];
    let truth = brute_force(&a, &b);
    for seed in 0u64..4 {
        let out = run_until_complete(&a, &b, seed);
        assert_matches_truth(&out.support, &truth);
    }
}

#[test]
fn sparse_convolve_eps_matches_truth() {
    let a: Vec<(u64, u64)> = (0..6).map(|i| (i * 7 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..6).map(|i| (i * 11 + 3, 1)).collect();
    let truth = brute_force(&a, &b);
    for seed in 0u64..4 {
        let out = convolve_seeded_eps::<Goldilocks>(&a, &b, seed, 1.0, u32::MAX);
        assert!(out.termination() == Termination::Complete);
        assert_matches_truth(&out.support, &truth);
    }
}

#[test]
fn sparse_convolve_dense_collisions() {
    let a = vec![(1u64, 1), (2, 1), (3, 1)];
    let b = vec![(1u64, 1), (2, 1), (3, 1)];
    let truth = brute_force(&a, &b);
    for seed in 0u64..4 {
        let out = run_until_complete(&a, &b, seed);
        assert_matches_truth(&out.support, &truth);
    }
}

#[test]
fn sparse_convolve_large_indices() {
    let a = vec![(1u64 << 40, 1), ((1u64 << 40) + 7, 1)];
    let b = vec![(1u64 << 39, 1), ((1u64 << 39) + 3, 1)];
    let truth = brute_force(&a, &b);
    let out = run_until_complete(&a, &b, 0x00C0_FFEE);
    assert_matches_truth(&out.support, &truth);
}

#[test]
fn budget_zero_returns_empty_lower_bound() {
    let a: Vec<(u64, u64)> = (0..10).map(|i| (i + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..10).map(|i| (i + 100, 1)).collect();
    let out = convolve_seeded::<Goldilocks>(&a, &b, 0, 0);
    assert!(out.termination() == Termination::LowerBound);
    assert!(out.support.is_empty());
}

#[test]
fn budget_partial_is_lower_bound() {
    let a: Vec<(u64, u64)> = (0..10).map(|i| (i * 7 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..10).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);
    let out = convolve_seeded::<Goldilocks>(&a, &b, 0, 4);
    assert!(out.termination() == Termination::LowerBound);
    for &(idx, c) in &out.support {
        assert!(c <= *truth.get(&idx).unwrap_or(&0));
    }
}

#[test]
fn budget_unlimited_terminates_complete() {
    let a: Vec<(u64, u64)> = (0..6).map(|i| (i * 7 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..6).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);
    let out = convolve_seeded::<Goldilocks>(&a, &b, 0, u32::MAX);
    assert!(out.termination() == Termination::Complete);
    assert_matches_truth(&out.support, &truth);
}

#[test]
fn budget_eps_partial_is_lower_bound() {
    let a: Vec<(u64, u64)> = (0..10).map(|i| (i * 7 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..10).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);
    let out = convolve_seeded_eps::<Goldilocks>(&a, &b, 0, 1.0, 3);
    assert!(out.termination() == Termination::LowerBound);
    for &(idx, c) in &out.support {
        assert!(c <= *truth.get(&idx).unwrap_or(&0));
    }
}

#[test]
fn is_prime_known_primes_and_composites() {
    for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 997, 998_244_353, 1_000_003] {
        assert!(is_prime(p), "expected prime: {p}");
    }
    for &n in &[0u64, 1, 4, 6, 9, 25, 49, 121, 1024, 1_000_000] {
        assert!(!is_prime(n), "expected composite: {n}");
    }
}

#[test]
fn sample_prime_in_returns_prime() {
    let mut rng = SplitMix::new(42);
    for _ in 0..16 {
        let p = sample_prime_in(100, 200, &mut rng);
        assert!((100..=200).contains(&p));
        assert!(is_prime(p));
    }
}

#[test]
fn alg5_recovers_residual_when_c_empty() {
    let a = vec![(2u64, 1u64), (5, 1), (11, 1)];
    let b = vec![(3u64, 1u64), (7, 1)];
    let truth = brute_force(&a, &b);

    let p = 31u64;
    let mut acc: HashMap<u64, u64> = HashMap::new();
    let no_c: [(u64, u64); 0] = [];
    for (idx, c) in bucketed_recover_residual::<Goldilocks, _>(&a, &b, no_c, p) {
        let entry = acc.entry(idx).or_insert(0);
        *entry = (*entry).max(c);
    }
    assert!(!acc.is_empty());
    for (&idx, &c) in &truth {
        assert!(acc.get(&idx).copied().unwrap_or(0) <= c, "idx={idx}");
    }
}

#[test]
fn alg5_subtracts_c() {
    // C = A⋆B = {5: 1} ⇒ residual is zero everywhere.
    let inputs_a = vec![(2u64, 1u64)];
    let inputs_b = vec![(3u64, 1u64)];
    let inputs_c = [(5u64, 1u64)];
    let prime = 7u64;
    let residual = bucketed_recover_residual::<Goldilocks, _>(
        &inputs_a,
        &inputs_b,
        inputs_c.iter().copied(),
        prime,
    );
    assert!(residual.is_empty(), "residual should be empty when C = A⋆B");
}

#[test]
fn log2_ceil_corner_cases() {
    assert_eq!(log2_ceil(0), 0);
    assert_eq!(log2_ceil(1), 0);
    assert_eq!(log2_ceil(2), 1);
    assert_eq!(log2_ceil(3), 2);
    assert_eq!(log2_ceil(4), 2);
    assert_eq!(log2_ceil(5), 3);
    assert_eq!(log2_ceil(8), 3);
    assert_eq!(log2_ceil(9), 4);
    assert_eq!(log2_ceil(1u64 << 63), 63);
}

#[test]
fn hybrid_empty_returns_complete_empty() {
    let out = convolve_seeded_hybrid::<Goldilocks>(&[], &[(1, 1)], 0, u32::MAX);
    assert!(out.termination() == Termination::Complete && out.support.is_empty());
}

#[test]
fn hybrid_single_pair() {
    let out = convolve_seeded_hybrid::<Goldilocks>(&[(2, 1)], &[(3, 1)], 42, u32::MAX);
    assert!(out.termination() == Termination::Complete);
    assert_eq!(out.support, vec![(5, 1)]);
}

#[test]
fn hybrid_matches_truth_boolean() {
    let a: Vec<(u64, u64)> = (0..6).map(|i| (i * 3 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..5).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);
    for seed in 0u64..3 {
        let out = convolve_seeded_hybrid::<Goldilocks>(&a, &b, seed, u32::MAX);
        assert!(out.termination() == Termination::Complete);
        assert_matches_truth(&out.support, &truth);
    }
}

#[test]
fn hybrid_matches_truth_weighted() {
    let a = vec![(2u64, 3u64), (5, 1), (11, 2)];
    let b = vec![(7u64, 4u64), (13, 5)];
    let truth = brute_force(&a, &b);
    for seed in 0u64..3 {
        let out = convolve_seeded_hybrid::<Goldilocks>(&a, &b, seed, u32::MAX);
        assert!(out.termination() == Termination::Complete);
        assert_matches_truth(&out.support, &truth);
    }
}

#[test]
fn hybrid_budget_partial_is_lower_bound() {
    let a: Vec<(u64, u64)> = (0..10).map(|i| (i * 7 + 1, 1)).collect();
    let b: Vec<(u64, u64)> = (0..10).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&a, &b);
    let out = convolve_seeded_hybrid::<Goldilocks>(&a, &b, 0, 5);
    assert!(out.termination() == Termination::LowerBound);
    for &(idx, c) in &out.support {
        assert!(c <= *truth.get(&idx).unwrap_or(&0), "idx={idx} c={c}");
    }
}

#[test]
fn within_call_phi_split_does_not_break_lower_bound() {
    // z=4 has pairs (1,3),(2,2),(3,1) that can split across φ; lower bound must hold.
    let a = vec![(1u64, 1), (2, 1), (3, 1)];
    let b = vec![(1u64, 1), (2, 1), (3, 1)];
    let truth = brute_force(&a, &b);
    for seed in 0u64..100 {
        let h = LinearHash::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1, 8);
        let r = bucketed_recover::<Goldilocks>(&a, &b, &h);
        assert!(lower_bound_holds(&r, &truth), "seed={seed}");
    }
}

#[test]
fn alg1_completion_implies_strict_l1_match() {
    // complete=true ⇒ Σcounts = ‖A‖₁·‖B‖₁ AND pointwise match across multiple m-blocks.
    let a: Vec<(u64, u64)> = (0..6).map(|i| (i * 7 + 1, 2)).collect();
    let b: Vec<(u64, u64)> = (0..6).map(|i| (i * 5 + 3, 3)).collect();
    let truth = brute_force(&a, &b);
    let target: u128 = u128::from(sum_counts(&a)) * u128::from(sum_counts(&b));
    for seed in 0u64..4 {
        let out = convolve_seeded::<Goldilocks>(&a, &b, seed, u32::MAX);
        assert!(out.termination() == Termination::Complete);
        let total: u128 = out.support.iter().map(|&(_, c)| u128::from(c)).sum();
        assert_eq!(total, target, "seed={seed}: Σcounts ≠ ‖A‖₁·‖B‖₁");
        let actual: HashMap<u64, u64> = out.support.into_iter().collect();
        assert_eq!(actual, truth, "seed={seed}: support diverges from truth");
    }
}

/// Lemma 16: `Pr[R_z < (A⋆B)_z] ≤ c·t/m`; check empirical rate < 2·t/m.
#[test]
fn lemma_16_per_index_failure_rate_bounded() {
    let inputs_a: Vec<(u64, u64)> = (0..6).map(|i| (i * 7 + 1, 1)).collect();
    let inputs_b: Vec<(u64, u64)> = (0..6).map(|i| (i * 5 + 2, 1)).collect();
    let truth = brute_force(&inputs_a, &inputs_b);
    let truth_len = f64::from(u32::try_from(truth.len()).expect("test t fits u32"));
    let bucket_count: usize = 256;
    let trials = 500u64;
    let (mut failures, mut tested) = (0u64, 0u64);
    for seed in 0..trials {
        let hasher = LinearHash::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1, bucket_count);
        let recovered: HashMap<u64, u64> =
            bucketed_recover::<Goldilocks>(&inputs_a, &inputs_b, &hasher)
                .into_iter()
                .collect();
        for (&z, &truth_count) in &truth {
            if recovered.get(&z).copied().unwrap_or(0) < truth_count {
                failures += 1;
            }
            tested += 1;
        }
    }
    let rate = f64::from(u32::try_from(failures).expect("test failure count fits u32"))
        / f64::from(u32::try_from(tested).expect("test tested count fits u32"));
    let bound = 2.0 * truth_len
        / f64::from(u32::try_from(bucket_count).expect("test bucket_count fits u32"));
    assert!(
        rate < bound,
        "failure rate {rate:.4} exceeds 2·t/m = {bound:.4}"
    );
}

#[test]
fn bucketed_recover_phi_split_aggregates_to_truth() {
    // z=4 has 3 preimages; merge_max over seeds must reach 3 without double-counting.
    let a = vec![(1u64, 1), (2, 1), (3, 1)];
    let b = vec![(1u64, 1), (2, 1), (3, 1)];
    let truth = brute_force(&a, &b);
    let target_z = 4u64;
    let expected = *truth.get(&target_z).unwrap();
    let mut acc: HashMap<u64, u64> = HashMap::new();
    for seed in 0u64..256 {
        let h = LinearHash::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1, 16);
        let r = bucketed_recover::<Goldilocks>(&a, &b, &h);
        for (idx, c) in r {
            let entry = acc.entry(idx).or_insert(0);
            *entry = (*entry).max(c);
        }
    }
    assert_eq!(
        acc.get(&target_z).copied().unwrap_or(0),
        expected,
        "merge_max over many seeds must reach truth at z={target_z}"
    );
}


proptest! {
    #[test]
    fn prop_bucketed_recover_is_lower_bound(
        a in proptest::collection::vec((0u64..1000, 1u64..10), 0..8),
        b in proptest::collection::vec((0u64..1000, 1u64..10), 0..8),
        seed: u64,
        log_m in 1u32..=8u32,
    ) {
        let h = LinearHash::new(seed | 1, 1usize << log_m);
        let r = bucketed_recover::<Goldilocks>(&a, &b, &h);
        let truth = brute_force(&a, &b);
        prop_assert!(lower_bound_holds(&r, &truth));
    }

    #[test]
    fn prop_sparse_convolve_matches_brute_force(
        a in proptest::collection::vec((0u64..100, 1u64..5), 0..6),
        b in proptest::collection::vec((0u64..100, 1u64..5), 0..6),
        seed: u64,
    ) {
        let truth = brute_force(&a, &b);
        let out = convolve_seeded::<Goldilocks>(&a, &b, seed, u32::MAX);
        prop_assume!(out.termination() == Termination::Complete);
        let actual: HashMap<u64, u64> = out.support.into_iter().collect();
        prop_assert_eq!(actual.len(), truth.len());
        for (idx, &count) in &truth {
            prop_assert_eq!(actual.get(idx).copied().unwrap_or(0), count);
        }
    }

    #[test]
    fn prop_alg1_and_alg4_agree_when_complete(
        a in proptest::collection::vec((0u64..50, 1u64..3), 1..5),
        b in proptest::collection::vec((0u64..50, 1u64..3), 1..5),
        seed: u64,
    ) {
        let alg1 = convolve_seeded::<Goldilocks>(&a, &b, seed, u32::MAX);
        let alg4 = convolve_seeded_eps::<Goldilocks>(&a, &b, seed, 1.0, u32::MAX);
        prop_assume!(alg1.termination() == Termination::Complete && alg4.termination() == Termination::Complete);
        let m1: HashMap<u64, u64> = alg1.support.into_iter().collect();
        let m4: HashMap<u64, u64> = alg4.support.into_iter().collect();
        prop_assert_eq!(m1, m4);
    }

    #[test]
    fn prop_alg5_residual_is_lower_bound(
        a in proptest::collection::vec((0u64..200, 1u64..3), 1..5),
        b in proptest::collection::vec((0u64..200, 1u64..3), 1..5),
        keep_fraction in 0.0f64..=1.0,
        p_choice in 0usize..6,
    ) {
        // C built as a subset of A⋆B so the precondition (A⋆B - C ≥ 0) holds.
        let truth = brute_force(&a, &b);
        let mut c: Vec<(u64, u64)> = truth
            .iter()
            .filter(|(idx, _)| (idx.wrapping_mul(2_654_435_761) as f64 / u64::MAX as f64).abs() < keep_fraction)
            .map(|(&k, &v)| (k, v))
            .collect();
        c.sort_unstable();
        let primes = [7u64, 11, 13, 17, 19, 23];
        let p = primes[p_choice];
        let r = bucketed_recover_residual::<Goldilocks, _>(&a, &b, c.iter().copied(), p);
        let c_map: HashMap<u64, u64> = c.into_iter().collect();
        for (idx, count) in r {
            let truth_count = truth.get(&idx).copied().unwrap_or(0);
            let c_count = c_map.get(&idx).copied().unwrap_or(0);
            prop_assert!(count <= truth_count.saturating_sub(c_count));
        }
    }

    #[test]
    fn prop_hybrid_matches_brute_force(
        a in proptest::collection::vec((0u64..50, 1u64..3), 1..4),
        b in proptest::collection::vec((0u64..50, 1u64..3), 1..4),
        seed: u64,
    ) {
        let truth = brute_force(&a, &b);
        let out = convolve_seeded_hybrid::<Goldilocks>(&a, &b, seed, u32::MAX);
        prop_assume!(out.termination() == Termination::Complete);
        let actual: HashMap<u64, u64> = out.support.into_iter().collect();
        prop_assert_eq!(actual.len(), truth.len());
        for (idx, &count) in &truth {
            prop_assert_eq!(actual.get(idx).copied().unwrap_or(0), count);
        }
    }
}
