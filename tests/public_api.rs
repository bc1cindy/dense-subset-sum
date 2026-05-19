//! Integration tests for the public API of `dense-subset-sum`.
//! Treats the crate as an external dependency would (no `crate::` paths).

use dense_subset_sum::count::companion::{log_w_signed, sasamoto_approx};
use dense_subset_sum::count::sparse_conv::Goldilocks;
use dense_subset_sum::count::sumset::GradedSumsetBudget;
use dense_subset_sum::{
    Bound, Bracket, GradedSumset, KNEE, Regime, Transaction, brute_force_w, dp_w, fixtures,
    log_w_for_e_sat,
};

// GradedSumset queries: small-N exactness, large-N truncation.
mod sparse_conv_lookup {
    use super::*;

    #[test]
    fn small_n_returns_exact_counts() {
        let a: Vec<u64> = (1..=5).collect();
        let s: GradedSumset = GradedSumset::bounded(&a, &[5, 6, 15], a.len());
        // E=5: subsets {5}, {1,4}, {2,3} → 3 confirmed
        assert_eq!(s.count_total(5).visible(), 3);
        // E=6: subsets {1,5}, {2,4}, {1,2,3} → 3
        assert_eq!(s.count_total(6).visible(), 3);
        // E=15: {1,2,3,4,5} only
        assert_eq!(s.count_total(15).visible(), 1);
        assert_eq!(s.bound_total(), Bound::Exact);
    }

    #[test]
    fn w_m_e_decomposes_w_e() {
        let a: Vec<u64> = (1..=6).collect();
        let s: GradedSumset = GradedSumset::bounded(&a, &[7], a.len());
        let total = s.count_total(7).visible();
        let sum_at: u32 = (0..=a.len()).map(|m| s.count_at(m, 7).visible()).sum();
        assert_eq!(total, sum_at, "W(E) == Σ_m W(M,E)");
    }

    #[test]
    fn large_n_saturates_with_lower_bound_flag() {
        // 25 ones, E=5 reachable by {1,1,1,1,1} within KNEE=5. Higher degrees truncated.
        let a = vec![1u64; 25];
        let s: GradedSumset = GradedSumset::bounded(&a, &[5], KNEE);
        let c = s.count_total(5);
        assert_eq!(
            c.bound(),
            Bound::LowerBound,
            "max_degree < N must mark lower-bound"
        );
        assert!(c.visible() > 0, "E=5 reachable within KNEE=5");
    }

    #[test]
    fn count_total_u32_has_headroom_for_large_counts() {
        // 30 small inputs with many low-degree paths to E=2.
        let a = vec![1u64; 30];
        let s: GradedSumset = GradedSumset::bounded(&a, &[2], 2);
        // C(30,2)=435 fits u32 confirmed.
        let total = s.count_total(2);
        assert!(matches!(
            total,
            dense_subset_sum::Count::Confirmed(435) | dense_subset_sum::Count::Truncated(_)
        ));
    }
}

// brute_force_w must equal dp_w on small inputs.
mod oracles {
    use super::*;

    #[test]
    fn brute_force_matches_dp() {
        let a: Vec<u64> = (1..=8).collect();
        for e in 0..=36 {
            let bf = brute_force_w(&a, e).unwrap();
            let dp = dp_w(&a, e, 1_000_000).unwrap();
            assert_eq!(bf, dp, "mismatch at E={}", e);
        }
    }

    #[test]
    fn dp_restricted_matches_per_degree_in_brute_force() {
        use dense_subset_sum::brute_force_w_restricted;
        use dense_subset_sum::dp_w_restricted;
        let a: Vec<u64> = (1..=6).collect();
        for m in 0..=a.len() {
            for e in 0..=21 {
                let bf = brute_force_w_restricted(&a, m, e).unwrap();
                let dp = dp_w_restricted(&a, m, e, 1_000_000).unwrap();
                assert_eq!(bf, dp, "M={}, E={}", m, e);
            }
        }
    }

    #[test]
    fn sparse_conv_is_lower_bound_of_full_w() {
        // sparse_conv (production path) must never exceed brute_force_w.
        let a: Vec<u64> = (1..=8).collect();
        for e in [3, 8, 15, 25] {
            let bf = brute_force_w(&a, e).unwrap();
            let s: GradedSumset = GradedSumset::bounded(&a, &[e], a.len());
            let sc = s.count_total(e).visible() as u128;
            assert!(sc <= bf, "E={}: sparse_conv {} > truth {}", e, sc, bf);
        }
    }
}

// Sasamoto saddle-point gated by Bracket::Dense.
mod sasamoto_companion {
    use super::*;

    #[test]
    fn log_w_for_e_sat_returns_finite_for_feasible_target() {
        let a: Vec<u64> = (1..=20).collect();
        let lw = log_w_for_e_sat(&a, 100);
        assert!(lw.is_finite(), "feasible E should return finite log W");
    }

    #[test]
    fn bracket_classifies_regime() {
        // Bracket regime should not be Dense for uniform inputs at MAX_MONEY/N anchoring;
        // assertion documents the gate behavior (Sparse/Transitional for Bitcoin-scale).
        let a = vec![1u64; 100];
        let bracket = Bracket::new(a.iter().copied(), 25).expect("valid bracket");
        assert!(
            matches!(bracket.regime(), Regime::Sparse | Regime::Transitional),
            "uniform inputs anchored at MAX_MONEY/N are not Dense; got {:?}",
            bracket.regime()
        );
    }

    #[test]
    fn sasamoto_approx_gated_by_bracket() {
        // Sparse regime: a single large input dominates.
        let a = vec![1_000_000, 1, 1, 1];
        let result = sasamoto_approx(&a, 500_000);
        assert!(
            result.is_none() || result.unwrap().is_finite(),
            "sparse regime should return None or finite gate"
        );
    }

    #[test]
    fn sasamoto_approx_close_to_exact_at_n20() {
        // N=20, balanced uniform inputs, mid-range E: Dense regime expected.
        let a: Vec<u64> = (1..=20).collect();
        let e = 100u64;
        let exact = brute_force_w(&a, e).unwrap() as f64;
        let approx = sasamoto_approx(&a, e);
        if let Some(la) = approx {
            let lex = exact.ln();
            let rel_err = (la - lex).abs() / lex.abs();
            assert!(
                rel_err < 0.10,
                "Sasamoto deviates {:.1}% from exact",
                rel_err * 100.0
            );
        }
    }
}

// Fee-aware multiset balance via log_w_signed.
mod signed_probe {
    use super::*;

    #[test]
    fn log_w_signed_balanced_tx_finite() {
        let tx = fixtures::maurer_fig2();
        let r = log_w_signed::<Goldilocks>(
            &tx.inputs,
            &tx.outputs,
            tx.fee(),
            KNEE,
            GradedSumsetBudget::default(),
        );
        assert!(r.is_ok(), "balanced tx should produce finite signed log_w");
    }

    #[test]
    fn log_w_signed_empty_sides_returns_empty_input() {
        use dense_subset_sum::SignedError;
        let r = log_w_signed::<Goldilocks>(&[], &[], 0, KNEE, GradedSumsetBudget::default());
        assert_eq!(r, Err(SignedError::EmptyInput));
    }

    #[test]
    fn log_w_signed_unreachable_target_is_err_unreachable() {
        use dense_subset_sum::SignedError;
        // fee much larger than any input combination can reach.
        let r = log_w_signed::<Goldilocks>(
            &[1, 1],
            &[1, 1],
            1_000_000,
            KNEE,
            GradedSumsetBudget::default(),
        );
        assert_eq!(r, Err(SignedError::Unreachable));
    }
}

// Wasabi 2 fixture cjtxs through per-coin probe.
mod wasabi2_pipeline {
    use super::*;
    use dense_subset_sum::harness::vs_cja::per_coin_measurements;

    fn fraction_reachable(
        measurements: &[dense_subset_sum::harness::vs_cja::CoinMeasurement],
    ) -> f64 {
        let n = measurements.len();
        if n == 0 {
            return 0.0;
        }
        let r = measurements
            .iter()
            .filter(|c| c.log_w_signed.is_some())
            .count();
        r as f64 / n as f64
    }

    #[test]
    fn all_positive_fixtures_compute_per_coin() {
        let txs = fixtures::all_wasabi2_positive_cjtxs();
        assert_eq!(txs.len(), 30, "expected 30 positive fixtures");
        for (label, tx) in &txs {
            let m = per_coin_measurements(tx, KNEE);
            assert_eq!(m.len(), tx.inputs.len() + tx.outputs.len(), "{label}");
            let frac = fraction_reachable(&m);
            assert!(frac > 0.1, "{label}: only {frac:.2} of coins reachable");
        }
    }

    #[test]
    fn all_false_fixtures_compute_per_coin() {
        let txs = fixtures::all_wasabi2_false_cjtxs();
        assert_eq!(txs.len(), 20, "expected 20 false fixtures");
        for (label, tx) in &txs {
            let m = per_coin_measurements(tx, KNEE);
            assert_eq!(m.len(), tx.inputs.len() + tx.outputs.len(), "{label}");
        }
    }

    #[test]
    fn graded_sumset_handles_real_wasabi2_inputs() {
        // Confirm sparse convolution runs on a real Wasabi2 CoinJoin.
        let (label, tx) = fixtures::all_wasabi2_positive_cjtxs()
            .into_iter()
            .find(|(l, _)| *l == "w2pos_03b4bd61_20in34out")
            .expect("fixture present");
        let s: GradedSumset = GradedSumset::bounded(&tx.inputs, &tx.outputs, KNEE);
        let any_hit = tx.outputs.iter().any(|&o| s.count_total(o).visible() > 0);
        assert!(any_hit, "{label}: no output reachable from input sumset");
    }
}

// Large-N synthetic: stress sparse_conv saturation and Sasamoto.
mod large_n {
    use super::*;

    #[test]
    fn n400_critical_path_returns_lower_bound() {
        let a: Vec<u64> = (1..=400).collect();
        let s: GradedSumset = GradedSumset::bounded(&a, &[200], KNEE);
        let c = s.count_total(200);
        assert_eq!(c.bound(), Bound::LowerBound, "N=400 must saturate");
        assert!(c.visible() > 0, "feasible E=200 should be reachable");
    }

    #[test]
    fn n400_signed_probe_unconstrained() {
        let inputs: Vec<u64> = (1..=400).collect();
        let outputs = inputs.clone();
        let r =
            log_w_signed::<Goldilocks>(&inputs, &outputs, 0, KNEE, GradedSumsetBudget::default());
        assert!(r.is_ok(), "N=400 should still resolve under signed probe");
    }

    #[test]
    fn n500_critical_path_returns_lower_bound() {
        let a: Vec<u64> = (1..=500).collect();
        let s: GradedSumset = GradedSumset::bounded(&a, &[300], KNEE);
        let c = s.count_total(300);
        assert_eq!(c.bound(), Bound::LowerBound);
        assert!(c.visible() > 0);
    }

    #[test]
    fn n500_brute_force_rejects_too_large() {
        let a: Vec<u64> = (1..=500).collect();
        let r = brute_force_w(&a, 250);
        assert!(r.is_err(), "N=500 must exceed brute-force budget");
    }

    #[test]
    fn n500_sasamoto_raw_returns_finite_positive() {
        // `sasamoto_approx` gate is conservative (typically None at Bitcoin scale);
        // raw `log_w_for_e_sat` path always returns the asymptotic, caller decides.
        let a: Vec<u64> = (1..=500).collect();
        let e: u64 = a.iter().sum::<u64>() / 2;
        let lw = log_w_for_e_sat(&a, e);
        assert!(
            lw.is_finite() && lw > 0.0,
            "N=500 saddle-point must be positive finite"
        );
    }

    #[test]
    fn n1000_critical_path_returns_lower_bound() {
        let a: Vec<u64> = (1..=1000).collect();
        let s: GradedSumset = GradedSumset::bounded(&a, &[600], KNEE);
        let c = s.count_total(600);
        assert_eq!(c.bound(), Bound::LowerBound);
        assert!(c.visible() > 0);
    }

    #[test]
    fn n1000_sasamoto_raw_returns_finite_positive() {
        let a: Vec<u64> = (1..=1000).collect();
        let e: u64 = a.iter().sum::<u64>() / 2;
        let lw = log_w_for_e_sat(&a, e);
        assert!(lw.is_finite() && lw > 0.0);
    }

    #[test]
    fn n2000_sasamoto_raw_returns_finite_positive() {
        let a: Vec<u64> = (1..=2000).collect();
        let e: u64 = a.iter().sum::<u64>() / 2;
        let lw = log_w_for_e_sat(&a, e);
        assert!(
            lw.is_finite() && lw > 0.0,
            "N=2000 saddle-point must be positive finite"
        );
    }
}

// Sparse_conv and Sasamoto must track log2(non_derived_mappings) for N+M <= 18.
mod boltzmann_validation {
    use super::*;
    use dense_subset_sum::harness::vs_cja::{
        FeeHandling, MappingComparison, compare_w_vs_mappings_with, correlate_w_vs_mappings,
    };

    fn small_test_set() -> Vec<(&'static str, Transaction)> {
        // Fixtures chosen to span n_non_derived from 1 (deterministic) to many
        // (high-ambiguity). Pure equal-denom txs collapse to n=1; mixed-denom
        // txs with arithmetic relations produce multiple non-derived partitions.
        vec![
            (
                "unambiguous_2x2",
                Transaction::new(vec![100, 200], vec![100, 200]),
            ),
            ("maurer_fig2", fixtures::maurer_fig2()),
            ("permuted_3", Transaction::new(vec![1, 2, 3], vec![3, 2, 1])),
            (
                "arith_4",
                Transaction::new(vec![1, 2, 3, 4], vec![1, 2, 3, 4]),
            ),
            (
                "arith_5",
                Transaction::new(vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 5]),
            ),
            ("eq_denominations", fixtures::equal_denominations()),
            (
                "split_3pairs_to_3sums",
                Transaction::new(vec![3, 3, 3, 3, 3, 3], vec![6, 6, 6]),
            ),
            (
                "merge_pairs_to_singles",
                Transaction::new(vec![5, 5, 5, 5], vec![10, 10]),
            ),
        ]
    }

    /// Probe-only: prints n_non_derived per fixture to inform threshold choice.
    /// Run with `cargo test boltzmann_validation::probe -- --include-ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_n_non_derived_per_fixture() {
        for (label, tx) in small_test_set() {
            let mc = compare_w_vs_mappings_with(&tx, label, KNEE, 18, FeeHandling::PhantomOutput);
            match mc {
                Some(mc) => println!(
                    "  {:30}  N+M={:2}  n_mappings={:5}  n_non_derived={:5}  H={:.2}  max_log_w_lookup={:.2}",
                    mc.label,
                    mc.total_coins,
                    mc.n_mappings,
                    mc.n_non_derived,
                    mc.entropy,
                    mc.max_log_w_lookup,
                ),
                None => println!("  {:30}  SKIPPED (> max_coins)", label),
            }
        }
    }

    fn compare_all() -> Vec<MappingComparison> {
        small_test_set()
            .iter()
            .filter_map(|(label, tx)| {
                compare_w_vs_mappings_with(tx, label, KNEE, 18, FeeHandling::PhantomOutput)
            })
            .collect()
    }

    /// One e2e test for the full Boltzmann pipeline: enumerates mappings,
    /// runs sparse_conv lookup + Sasamoto per sub-tx, asserts correlation
    /// with `ln(n_non_derived + 1)` across the fixture range.
    #[test]
    fn sparse_conv_tracks_boltzmann_across_small_txs() {
        let comparisons = compare_all();
        assert_eq!(comparisons.len(), 8, "all fixtures fit max_coins=18");

        for mc in &comparisons {
            match mc.label.as_str() {
                "unambiguous_2x2" | "maurer_fig2" => {
                    assert_eq!(mc.n_non_derived, 1, "{}: unique interpretation", mc.label);
                    assert_eq!(mc.entropy, 0.0);
                }
                "eq_denominations" => {
                    assert!(
                        mc.n_non_derived >= 10 && mc.entropy >= 3.0,
                        "{}: high entropy expected, got n={}, H={:.2}",
                        mc.label,
                        mc.n_non_derived,
                        mc.entropy
                    );
                }
                _ => {}
            }
        }

        // `correlate_w_vs_mappings` filters fixtures with n=0 or non-finite lookup.
        let corr =
            correlate_w_vs_mappings(&comparisons).expect("≥3 finite samples after filtering");
        assert!(
            corr.n_transactions >= 5,
            "need ≥5 finite samples post-filter, got {}",
            corr.n_transactions
        );
        assert!(
            corr.spearman_lookup > 0.6,
            "sparse_conv lookup must rank-order with Boltzmann: ρ={:.3}",
            corr.spearman_lookup
        );
        assert!(
            corr.pearson_lookup > 0.6,
            "linear-in-log-space expected: Pearson={:.3}",
            corr.pearson_lookup
        );

        let max_n = comparisons.iter().map(|m| m.n_non_derived).max().unwrap();
        let max_lw = comparisons
            .iter()
            .filter(|m| m.n_non_derived == max_n)
            .map(|m| m.max_log_w_lookup)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_lw > 0.5,
            "max-entropy fixture (n={}) should give max_log_w > 0.5, got {:.3}",
            max_n,
            max_lw
        );
    }
}
