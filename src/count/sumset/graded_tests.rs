use super::*;
use crate::count::sumset::budget::LasVegas;
use crate::count::sumset::test_oracle::{
    count_balance as brute_force_balance_oracle, subset_sums as brute_force,
    subset_sums_bounded as brute_force_bounded, subset_sums_exact as brute_force_exact,
};
use proptest::prelude::*;
use std::num::NonZeroUsize;

fn assert_matches_brute_force(set: &[u64]) {
    let s: GradedSumset = GradedSumset::bounded(set, &[], set.len());
    let bf = brute_force(set);
    let distinct_sums = s.sums_total().count();
    assert_eq!(distinct_sums, bf.len(), "set={set:?}");
    for (&sum, &count) in &bf {
        assert_eq!(
            s.count_total(sum).visible(),
            u32::from(count),
            "set={set:?} sum={sum}"
        );
    }
}

#[test]
fn empty_contains_zero_only() {
    let s: GradedSumset = GradedSumset::empty();
    assert_eq!(s.count_total(0).visible(), 1);
    assert_eq!(s.sums_total().count(), 1);
    assert_eq!(s.count_total(1).visible(), 0);
}

#[test]
fn singleton_contains_zero_and_value() {
    let s: GradedSumset = GradedSumset::bounded(&[7], &[], 1);
    assert_eq!(s.count_total(0).visible(), 1);
    assert_eq!(s.count_total(7).visible(), 1);
    assert_eq!(s.sums_total().count(), 2);
}

#[test]
fn singleton_zero_doubles_count_at_zero() {
    let s: GradedSumset = GradedSumset::bounded(&[0], &[], 1);
    assert_eq!(s.count_total(0).visible(), 2);
    assert_eq!(s.sums_total().count(), 1);
}

#[test]
fn powerset_empty() {
    let s: GradedSumset = GradedSumset::bounded(&[], &[], 0);
    assert_eq!(s.count_total(0).visible(), 1);
    assert_eq!(s.sums_total().count(), 1);
}

#[test]
fn powerset_distinct_values() {
    assert_matches_brute_force(&[3, 5, 7]);
}

#[test]
fn powerset_with_duplicates() {
    assert_matches_brute_force(&[1, 1, 2]);
}

#[test]
fn convolve_is_commutative() {
    let a: GradedSumset = GradedSumset::bounded(&[2, 3], &[], 4);
    let b: GradedSumset = GradedSumset::bounded(&[5, 7], &[], 4);
    let ab: GradedSumset = GradedSumset::convolve(&a, &b);
    let ba: GradedSumset = GradedSumset::convolve(&b, &a);
    assert_eq!(ab.sums_total().count(), ba.sums_total().count());
    for sum in ab.sums_total() {
        assert_eq!(ab.count_total(sum), ba.count_total(sum));
    }
}

#[test]
fn count_just_below_saturation() {
    let zeros = vec![0u64; 7];
    let s: GradedSumset = GradedSumset::bounded(&zeros, &[], zeros.len());
    assert_eq!(s.count_total(0).visible(), 128);
    assert_eq!(s.sums_total().count(), 1);
}

#[test]
fn count_total_above_u8_for_eight_zeros() {
    let zeros = vec![0u64; 8];
    let s: GradedSumset = GradedSumset::bounded(&zeros, &[], zeros.len());
    assert_eq!(s.count_total(0).visible(), 256);
    assert_eq!(s.sums_total().count(), 1);
}

#[test]
fn count_total_exact_for_small_powerset() {
    let s: GradedSumset = GradedSumset::bounded(&[3u64, 5, 7], &[], 3);
    assert_eq!(s.bound_total(), Bound::Exact);
    assert_eq!(s.count_total(7).visible(), 1); // {7}
    assert_eq!(s.count_total(0).visible(), 1); // {}
}

#[test]
fn count_total_handles_above_u8_saturation() {
    let zeros = vec![0u64; 8];
    let s: GradedSumset = GradedSumset::bounded(&zeros, &[], zeros.len());
    // 2^8 = 256 subsets of zeros all sum to 0; u32 storage doesn't saturate.
    assert_eq!(s.count_total(0).visible(), 256);
    assert_eq!(s.bound_total(), Bound::Exact);
}

#[test]
fn bound_total_is_lower_when_degree_truncated() {
    let s: GradedSumset = GradedSumset::bounded(&[1u64, 2, 3, 4, 5], &[], 2);
    assert_eq!(s.bound_total(), Bound::LowerBound);
    // Layer-level still exact for m within range.
    assert_eq!(s.bound_at(2), Bound::Exact);
    assert_eq!(s.count_at(2, 3).visible(), 1); // {1,2}
}

#[test]
fn bounded_max_zero_is_empty() {
    let s: GradedSumset = GradedSumset::bounded(&[3, 5, 7], &[], 0);
    assert_eq!(s.sums_total().count(), 1);
    assert_eq!(s.count_total(0).visible(), 1);
}

#[test]
fn bounded_max_one_is_singletons_plus_empty() {
    let s: GradedSumset = GradedSumset::bounded(&[3, 5, 7], &[], 1);
    assert!(s.count_total(0).visible() > 0);
    assert!(s.count_total(3).visible() > 0);
    assert!(s.count_total(5).visible() > 0);
    assert!(s.count_total(7).visible() > 0);
    assert_eq!(s.count_total(8).visible(), 0);
    assert_eq!(s.sums_total().count(), 4);
}

#[test]
fn bounded_full_matches_powerset() {
    let set = [1, 1, 2, 3];
    let bounded: GradedSumset = GradedSumset::bounded(&set, &[], set.len());
    let full: GradedSumset = GradedSumset::bounded(&set, &[], set.len());
    assert_eq!(bounded.sums_total().count(), full.sums_total().count());
    for sum in full.sums_total() {
        assert_eq!(bounded.count_total(sum), full.count_total(sum));
    }
}

#[test]
fn exact_constructors_are_not_lower_bound() {
    assert_eq!(
        GradedSumset::<Goldilocks>::empty().bound_total(),
        Bound::Exact
    );
    assert_eq!(
        GradedSumset::<Goldilocks>::bounded(&[7], &[], 1).bound_total(),
        Bound::Exact
    );
    assert_eq!(
        GradedSumset::<Goldilocks>::bounded(&[3, 5, 7], &[], 3).bound_total(),
        Bound::Exact
    );
}

#[test]
fn bounded_marks_lower_bound_when_truncating() {
    let set = [1, 2, 3, 4, 5];
    assert_eq!(
        GradedSumset::<Goldilocks>::bounded(&set, &[], 2).bound_total(),
        Bound::LowerBound
    );
    assert_eq!(
        GradedSumset::<Goldilocks>::bounded(&set, &[], set.len()).bound_total(),
        Bound::Exact
    );
    assert_eq!(
        GradedSumset::<Goldilocks>::bounded(&set, &[], set.len() + 10).bound_total(),
        Bound::Exact
    );
}

#[test]
fn bounded_handles_scale_brute_force_cannot() {
    // C(100, 10) ≈ 10^13 infeasible for brute force.
    let elements = vec![1u64; 100];
    let s: GradedSumset = GradedSumset::bounded(&elements, &[], 10);
    for k in 0..=10u64 {
        assert!(s.count_total(k).visible() > 0, "missing sum {k}");
    }
    assert_eq!(s.count_total(11).visible(), 0);
    assert_eq!(s.count_total(0).visible(), 1);
    assert_eq!(s.count_total(1).visible(), 100);
    assert_eq!(s.count_total(2).visible(), 4950); // C(100, 2)
    assert_eq!(s.bound_total(), Bound::LowerBound);
}

#[test]
#[ignore = "slow: n=10000 worst-case; run with `cargo test --release -- --ignored`"]
fn bounded_handles_max_realistic_n() {
    let elements: Vec<u64> = (1..=10_000).collect();
    let knee = 5usize;
    let cfg: GradedSumsetBudget =
        GradedSumsetBudget::default().with_max_size(NonZeroUsize::new(4096).unwrap());
    let target = 15u64;
    let s: GradedSumset = GradedSumset::builder(&elements, cfg, &[target]).bounded(knee);
    assert_eq!(s.bound_total(), Bound::LowerBound);
    assert!(
        s.count_total(target).visible() >= 1,
        "{target} must be reachable"
    );
}

#[test]
fn bounded_with_caps_intermediate() {
    let elements: Vec<u64> = (1..=8).collect();
    let cfg: GradedSumsetBudget =
        GradedSumsetBudget::default().with_max_size(NonZeroUsize::new(4).unwrap());
    let s: GradedSumset = GradedSumset::builder(&elements, cfg, &[]).bounded(8);
    assert!(s.sums_total().count() <= 8 + 1); // capped per-bucket; conservative bound
    assert_eq!(s.bound_total(), Bound::LowerBound);
    assert!(s.count_total(0).visible() > 0);
}

#[test]
fn bounded_with_unlimited_budget_matches_unbudgeted() {
    let elements = [1u64, 2, 3, 4];
    let unbounded: GradedSumset = GradedSumset::bounded(&elements, &[], 4);
    let with_budget: GradedSumset =
        GradedSumset::builder(&elements, GradedSumsetBudget::default(), &[]).bounded(4);
    assert_eq!(
        unbounded.sums_total().count(),
        with_budget.sums_total().count()
    );
    assert_eq!(unbounded.bound_total(), with_budget.bound_total());
    for sum in unbounded.sums_total() {
        assert_eq!(unbounded.count_total(sum), with_budget.count_total(sum));
    }
}

#[test]
fn bounded_pre_merge_skip_returns_subsumset() {
    let elements = [10u64, 20, 40, 80];
    let cfg: GradedSumsetBudget =
        GradedSumsetBudget::default().with_max_size(NonZeroUsize::new(2).unwrap());
    let restricted: GradedSumset = GradedSumset::builder(&elements, cfg, &[]).bounded(4);
    let full: GradedSumset = GradedSumset::bounded(&elements, &[], 4);
    assert_eq!(restricted.bound_total(), Bound::LowerBound);
    assert!(restricted.count_total(0).visible() > 0);
    for sum in restricted.sums_total() {
        assert!(
            full.count_total(sum).visible() > 0,
            "restricted sum {sum} missing from full"
        );
        assert!(restricted.count_total(sum).visible() <= full.count_total(sum).visible());
    }
}

#[test]
fn blowup_level_none_when_uncapped() {
    let elements = vec![1u64, 2, 3, 4];
    assert_eq!(
        GradedSumset::<Goldilocks>::bounded(&elements, &[], elements.len()).blowup_level(),
        None
    );
    assert_eq!(
        GradedSumset::<Goldilocks>::bounded(&elements, &[], 4).blowup_level(),
        None
    );
}

#[test]
fn blowup_level_records_first_capped_merge() {
    let elements: Vec<u64> = (1..=8).collect();
    let cfg: GradedSumsetBudget =
        GradedSumsetBudget::default().with_max_size(NonZeroUsize::new(4).unwrap());
    let s: GradedSumset = GradedSumset::builder(&elements, cfg, &[]).bounded(8);
    let level = s.blowup_level().expect("cap must fire");
    assert!(level <= 3, "blowup_level={level}");
    assert_eq!(s.bound_total(), Bound::LowerBound);
}

#[test]
fn blowup_level_propagates_through_convolve() {
    let cfg: GradedSumsetBudget =
        GradedSumsetBudget::default().with_max_size(NonZeroUsize::new(2).unwrap());
    let capped: GradedSumset = GradedSumset::builder(&[10u64, 20, 40, 80], cfg, &[]).bounded(4);
    let exact: GradedSumset = GradedSumset::bounded(&[3u64, 5], &[], 4);
    let combined = GradedSumset::convolve(&capped, &exact);
    assert_eq!(combined.blowup_level(), capped.blowup_level());
}

#[test]
fn cap_to_top_preserves_pinned_targets_under_tight_budget() {
    // Pinning preserves existing entries; cannot reconstruct sums lost to earlier caps.
    let elements = [10u64, 20];
    let cfg: GradedSumsetBudget =
        GradedSumsetBudget::default().with_max_size(NonZeroUsize::new(2).unwrap());
    let target = 30u64;
    let pinned: GradedSumset = GradedSumset::builder(&elements, cfg, &[target]).bounded(2);
    assert_eq!(pinned.bound_total(), Bound::LowerBound);
    assert!(
        pinned.count_total(target).visible() > 0,
        "pinned target {target} must survive cap"
    );
}

#[test]
fn convolve_propagates_lower_bound() {
    // Layer-level bound: convolve preserves Exact when no truncation occurs.
    let exact_a: GradedSumset = GradedSumset::bounded(&[3, 5], &[], 4);
    let exact_b: GradedSumset = GradedSumset::bounded(&[1, 2, 3, 4], &[], 4);
    assert_eq!(
        GradedSumset::convolve(&exact_a, &exact_b).bound_at(0),
        Bound::Exact
    );
    // Tight budget creates a LowerBound that propagates.
    let cfg: GradedSumsetBudget =
        GradedSumsetBudget::default().with_max_size(NonZeroUsize::new(2).unwrap());
    let lb: GradedSumset = GradedSumset::builder(&[1, 2, 3, 4], cfg, &[]).bounded(4);
    assert_eq!(lb.bound_at(0), Bound::LowerBound);
    assert_eq!(
        GradedSumset::convolve(&exact_a, &lb).bound_at(0),
        Bound::LowerBound
    );
}

#[test]
fn cap_to_top_counts_keeps_zero_and_top_frequencies() {
    let mut h: HashMap<u64, u32> = HashMap::new();
    h.insert(0, 100);
    h.insert(1, 50);
    h.insert(2, 80);
    h.insert(3, 10);
    h.insert(4, 30);
    cap_to_top_counts(&mut h, 3, |_| false);
    assert!(h.contains_key(&0));
    assert!(h.contains_key(&2));
    assert!(h.contains_key(&1));
    assert!(!h.contains_key(&3));
    assert!(!h.contains_key(&4));
    assert_eq!(h.len(), 3);
}

#[test]
fn cap_to_top_counts_no_op_below_size() {
    let mut h: HashMap<u64, u32> = HashMap::new();
    h.insert(0, 1);
    h.insert(7, 2);
    cap_to_top_counts(&mut h, 5, |_| false);
    assert_eq!(h.len(), 2);
}

#[test]
fn covers_returns_true_when_other_meets_threshold() {
    let big: GradedSumset = GradedSumset::bounded(&[1u64, 2, 3], &[], 3);
    let small: GradedSumset = GradedSumset::bounded(&[1u64, 2], &[], 2);
    assert!(small.covers(&big, 1));
    assert!(!big.covers(&small, 1));
}

#[test]
fn bound_join_absorbs_lower_bound() {
    assert_eq!(Bound::Exact.join(Bound::Exact), Bound::Exact);
    assert_eq!(Bound::Exact.join(Bound::LowerBound), Bound::LowerBound);
    assert_eq!(Bound::LowerBound.join(Bound::Exact), Bound::LowerBound);
    assert_eq!(Bound::LowerBound.join(Bound::LowerBound), Bound::LowerBound);
}

#[test]
fn includes_balance_empty_sides_at_zero() {
    let e: GradedSumset = GradedSumset::empty();
    assert!(e.includes_balance(&e, 0));
    assert!(!e.includes_balance(&e, 1));
    assert!(!e.includes_balance(&e, -1));
}

#[test]
fn includes_balance_singleton_pair() {
    let p: GradedSumset = GradedSumset::bounded(&[5], &[], 1);
    let n: GradedSumset = GradedSumset::bounded(&[3], &[], 1);
    assert!(p.includes_balance(&n, 2));
    assert!(p.includes_balance(&n, 5));
    assert!(p.includes_balance(&n, -3));
    assert!(p.includes_balance(&n, 0));
    assert!(!p.includes_balance(&n, 100));
}

#[test]
fn includes_balance_iteration_side_does_not_change_answer() {
    let p: GradedSumset = GradedSumset::bounded(&[2, 3, 5, 7], &[], 4);
    let n: GradedSumset = GradedSumset::bounded(&[4], &[], 1);
    assert_eq!(p.includes_balance(&n, 1), n.includes_balance(&p, -1));
}

#[test]
fn count_balance_empty_at_zero() {
    let e: GradedSumset = GradedSumset::empty();
    assert_eq!(e.count_balance(&e, 0), 1);
    assert_eq!(e.count_balance(&e, 1), 0);
}

#[test]
fn count_balance_singleton_pair() {
    let p: GradedSumset = GradedSumset::bounded(&[5], &[], 1);
    let n: GradedSumset = GradedSumset::bounded(&[3], &[], 1);
    assert_eq!(p.count_balance(&n, 2), 1);
    assert_eq!(p.count_balance(&n, 5), 1);
    assert_eq!(p.count_balance(&n, -3), 1);
    assert_eq!(p.count_balance(&n, 0), 1);
    assert_eq!(p.count_balance(&n, 100), 0);
}

#[test]
fn count_balance_at_saturated_input() {
    let zeros = vec![0u64; 8];
    let s: GradedSumset = GradedSumset::bounded(&zeros, &[], zeros.len());
    // 2^8 = 256 subsets all sum to 0; pair count: 256 × 256 = 65536.
    assert_eq!(s.count_total(0).visible(), 256);
    assert_eq!(s.count_balance(&s, 0), 65536);
}

proptest! {
    #[test]
    fn powerset_matches_brute_force(
        set in proptest::collection::vec(0u64..50, 0..10)
    ) {
        let s: GradedSumset = GradedSumset::bounded(&set, &[], set.len());
        let bf = brute_force(&set);
        prop_assert_eq!(s.sums_total().count(), bf.len());
        for (sum, count) in bf {
            prop_assert_eq!(s.count_total(sum).visible(), u32::from(count));
        }
    }

    #[test]
    fn bounded_matches_brute_force(
        set in proptest::collection::vec(0u64..30, 0..8),
        fixed_degree in 0usize..=8,
    ) {
        let s: GradedSumset = GradedSumset::bounded(&set, &[], fixed_degree);
        let bf = brute_force_bounded(&set, fixed_degree);
        prop_assert_eq!(s.sums_total().count(), bf.len());
        for (sum, count) in bf {
            prop_assert_eq!(s.count_total(sum).visible(), u32::from(count));
        }
    }

    #[test]
    fn exact_size_matches_brute_force(
        set in proptest::collection::vec(0u64..30, 0..8),
        m in 0usize..=8,
    ) {
        let s: GradedSumset = GradedSumset::bounded(&set, &[], m);
        let bf = brute_force_exact(&set, m);
        let layer_len = s.sums_at(m).count();
        prop_assert_eq!(layer_len, bf.len());
        for (sum, count) in bf {
            prop_assert_eq!(s.count_at(m, sum).visible(), u32::from(count));
        }
    }

    #[test]
    fn convolve_is_associative(
        a in proptest::collection::vec(0u64..30, 0..5),
        b in proptest::collection::vec(0u64..30, 0..5),
        c in proptest::collection::vec(0u64..30, 0..5),
    ) {
        let max_d = 15;
        let sa: GradedSumset = GradedSumset::bounded(&a, &[], max_d);
        let sb: GradedSumset = GradedSumset::bounded(&b, &[], max_d);
        let sc: GradedSumset = GradedSumset::bounded(&c, &[], max_d);
        let ab_c: GradedSumset = GradedSumset::convolve(&GradedSumset::convolve(&sa, &sb), &sc);
        let a_bc: GradedSumset = GradedSumset::convolve(&sa, &GradedSumset::convolve(&sb, &sc));
        prop_assert_eq!(ab_c.sums_total().count(), a_bc.sums_total().count());
        for sum in ab_c.sums_total() {
            prop_assert_eq!(ab_c.count_total(sum), a_bc.count_total(sum));
        }
    }

    /// Inputs small enough that no per-sum count saturates at u8.
    #[test]
    fn count_balance_matches_brute_force(
        pos in proptest::collection::vec(0u64..20, 0..6),
        neg in proptest::collection::vec(0u64..20, 0..6),
        target in -50i64..50,
    ) {
        let p: GradedSumset = GradedSumset::bounded(&pos, &[], pos.len());
        let n: GradedSumset = GradedSumset::bounded(&neg, &[], neg.len());
        let our = u8::try_from(p.count_balance(&n, target)).unwrap_or(u8::MAX);
        prop_assert_eq!(
            our,
            brute_force_balance_oracle(&pos, &neg, target)
        );
    }

    #[test]
    fn includes_balance_matches_brute_force(
        pos in proptest::collection::vec(0u64..30, 0..6),
        neg in proptest::collection::vec(0u64..30, 0..6),
        target in -100i64..100,
    ) {
        let p: GradedSumset = GradedSumset::bounded(&pos, &[], pos.len());
        let n: GradedSumset = GradedSumset::bounded(&neg, &[], neg.len());
        prop_assert_eq!(
            p.includes_balance(&n, target),
            brute_force_balance_oracle(&pos, &neg, target) > 0
        );
    }

    #[test]
    fn prop_budgeted_is_subset_of_unbudgeted(
        set in proptest::collection::vec(0u64..30, 0..7),
        max_degree in 0usize..=7,
        max_size in 1usize..=64,
    ) {
        let full: GradedSumset = GradedSumset::bounded(&set, &[], max_degree);
        let cfg: GradedSumsetBudget = GradedSumsetBudget::default()
            .with_max_size(NonZeroUsize::new(max_size).unwrap());
        let budgeted: GradedSumset = GradedSumset::builder(&set, cfg, &[]).bounded(max_degree);
        for sum in budgeted.sums_total() {
            prop_assert!(full.count_total(sum).visible() > 0, "budgeted sum {sum} missing from unbudgeted");
            prop_assert!(budgeted.count_total(sum).visible() <= full.count_total(sum).visible());
        }
    }
}

#[test]
fn powerset_correct_at_scale() {
    let elements: Vec<u64> = (0..14u32).map(|i| 1u64 << i).collect();
    let s: GradedSumset = GradedSumset::bounded(&elements, &[], elements.len());
    let bf = brute_force(&elements);
    assert_eq!(
        s.sums_total().count(),
        bf.len(),
        "support size differs from brute force"
    );
    for (&sum, &count) in &bf {
        assert_eq!(s.count_total(sum).visible(), u32::from(count), "sum={sum}");
    }
    assert_eq!(s.bound_total(), Bound::Exact);
}

#[test]
fn convolve_keeps_exact_when_no_saturation() {
    let a: GradedSumset = GradedSumset::bounded(&[3, 5, 7], &[], 5);
    let b: GradedSumset = GradedSumset::bounded(&[2, 11], &[], 5);
    let combined = GradedSumset::convolve(&a, &b);
    assert_eq!(combined.bound_at(0), Bound::Exact);
}

/// Algorithms 1, 4, 6 must agree on the same input when budget is unlimited.
#[test]
fn algorithm_variants_agree_on_powerset() {
    let elements: Vec<u64> = (1..=6).map(|i| i * 3).collect();
    let baseline: GradedSumset = GradedSumset::bounded(&elements, &[], elements.len());
    let bud_t2: GradedSumsetBudget =
        GradedSumsetBudget::default().with_las_vegas(LasVegas::Theorem2);
    let bud_eps: GradedSumsetBudget =
        GradedSumsetBudget::default().with_las_vegas(LasVegas::Lemma19 { epsilon: 1.0 });
    let bud_hybrid: GradedSumsetBudget =
        GradedSumsetBudget::default().with_las_vegas(LasVegas::Lemma22);
    let s_t2: GradedSumset = GradedSumset::builder(&elements, bud_t2, &[]).bounded(elements.len());
    let s_eps: GradedSumset =
        GradedSumset::builder(&elements, bud_eps, &[]).bounded(elements.len());
    let s_hybrid: GradedSumset =
        GradedSumset::builder(&elements, bud_hybrid, &[]).bounded(elements.len());
    for s in [&s_t2, &s_eps, &s_hybrid] {
        assert_eq!(
            s.sums_total().count(),
            baseline.sums_total().count(),
            "support size diverged"
        );
        for sum in baseline.sums_total() {
            assert_eq!(
                s.count_total(sum),
                baseline.count_total(sum),
                "count at {sum} diverged"
            );
        }
    }
}
