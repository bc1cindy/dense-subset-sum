//! W estimators vs ground truth. Exhaustive enumeration for small N,
//! Monte Carlo sampling (with timeout) for larger N.

mod core;
mod empirical;
mod monte_carlo;
mod report;

pub use core::{
    BatchRow, CompareMode, CompareRegime, ComparisonReport, ComparisonRow, EstimatorSummary,
    aggregate_reports, classify_regime, compare, compare_dp_ground_truth, uniform_random_set,
};
pub use empirical::{EmpiricalDistribution, random_coinjoin};
pub use monte_carlo::compare_monte_carlo;
pub use report::{print_batch_summary, print_report, print_report_csv};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_basic() {
        let a: Vec<u64> = (1..=16).collect();
        let report = compare(&a, 50, 10, 1_000_000, "sequential_16");
        assert!(!report.rows.is_empty());
        assert!(report.sasamoto.spearman > 0.95);
        for row in &report.rows {
            if let Some(wl) = row.w_lookup {
                assert!(
                    (wl as f64) <= row.w_exact,
                    "lookup should be lower bound: E={}, lookup={}, exact={}",
                    row.e_target,
                    wl,
                    row.w_exact
                );
            }
        }
        assert!(matches!(report.mode, CompareMode::Exhaustive));
    }

    #[test]
    fn test_compare_monte_carlo_tracks_exhaustive() {
        // At N=20, exhaustive is 1M subsets. MC with 500k samples should land
        // close enough that Sasamoto error and correlation are comparable.
        let a: Vec<u64> = (1..=20).collect();
        let exact = compare(&a, 200, 10, 1_000_000, "n20_exact");
        let mc = compare_monte_carlo(&a, 100, 10, 1_000_000, "n20_mc", 500_000, 60_000, 7);

        assert!(matches!(mc.mode, CompareMode::MonteCarlo { .. }));
        assert!(!mc.rows.is_empty(), "MC should produce at least some rows");

        // Sasamoto Spearman should be high against both ground truths.
        assert!(
            exact.sasamoto.spearman > 0.9,
            "exact Spearman={}",
            exact.sasamoto.spearman
        );
        assert!(
            mc.sasamoto.spearman > 0.9,
            "MC Spearman={} (noisier but still correlated)",
            mc.sasamoto.spearman
        );
    }

    #[test]
    fn test_compare_monte_carlo_timeout_respected() {
        // Request 10G samples with a 50 ms cap — must return in well under a second.
        let a: Vec<u64> = (1..=16).collect();
        let start = std::time::Instant::now();
        let report =
            compare_monte_carlo(&a, 1, 8, 1_000_000, "timeout_probe", 10_000_000_000, 50, 1);
        let elapsed = start.elapsed().as_millis();
        assert!(elapsed < 1000, "took {} ms, expected <1000", elapsed);
        match report.mode {
            CompareMode::MonteCarlo {
                timed_out,
                samples_drawn,
                ..
            } => {
                assert!(timed_out, "should have timed out");
                assert!(samples_drawn > 0, "should have drawn at least 1 sample");
            }
            _ => panic!("expected MonteCarlo mode"),
        }
    }

    #[test]
    fn test_compare_monte_carlo_timeout_zero_disables_wall_clock() {
        // timeout_ms=0 means no wall-clock limit: run all n_samples.
        let a: Vec<u64> = (1..=12).collect();
        let report = compare_monte_carlo(&a, 1, 8, 1_000_000, "no_timeout", 4_096, 0, 42);
        match report.mode {
            CompareMode::MonteCarlo {
                timed_out,
                samples_drawn,
                samples_requested,
            } => {
                assert!(!timed_out, "should not time out when timeout_ms=0");
                assert_eq!(samples_drawn, 4_096);
                assert_eq!(samples_requested, 4_096);
            }
            _ => panic!("expected MonteCarlo mode"),
        }
    }

    #[test]
    #[should_panic(expected = "n_samples=0 produces no estimate")]
    fn test_compare_monte_carlo_rejects_zero_samples() {
        let a: Vec<u64> = (1..=10).collect();
        let _ = compare_monte_carlo(&a, 1, 8, 1_000, "zero_samples", 0, 1_000, 0);
    }

    #[test]
    fn test_uniform_random_set_deterministic() {
        let a1 = uniform_random_set(10, 1000, 42);
        let a2 = uniform_random_set(10, 1000, 42);
        assert_eq!(a1, a2);

        let a3 = uniform_random_set(10, 1000, 43);
        assert_ne!(a1, a3);
    }

    /// Skips cleanly so `cargo test` runs without the 353 MB download.
    fn load_cja_or_skip(test_name: &str) -> Option<EmpiricalDistribution> {
        match EmpiricalDistribution::try_load_default_cja() {
            Some(d) => Some(d),
            None => {
                eprintln!(
                    "skip {}: {} not present (run scripts/fetch_cja_distribution.sh)",
                    test_name,
                    EmpiricalDistribution::DEFAULT_CJA_PATH
                );
                None
            }
        }
    }

    #[test]
    fn test_cja_distribution_bin_loads_and_samples() {
        let Some(dist) = load_cja_or_skip("test_cja_distribution_bin_loads_and_samples") else {
            return;
        };
        assert!(
            dist.cdf_len() > 1000,
            "CDF too small, likely mis-parsed: {}",
            dist.cdf_len()
        );
        let vals = dist.random_set(128, 42);
        assert_eq!(vals.len(), 128);
        for &v in &vals {
            assert!(v >= 1);
        }
        let unique: std::collections::HashSet<u64> = vals.iter().copied().collect();
        assert!(unique.len() >= 64, "too few unique draws: {}", unique.len());
    }

    #[test]
    fn test_random_coinjoin() {
        let Some(dist) = load_cja_or_skip("test_random_coinjoin") else {
            return;
        };
        let tx = random_coinjoin(&dist, 3, 2, 42);
        assert_eq!(tx.inputs.len(), 6);
        assert_eq!(tx.outputs.len(), 6);
        assert_eq!(tx.fee(), Some(0));
    }

    #[test]
    fn test_uniform_random_comparison() {
        let configs: Vec<(usize, u64)> = vec![(8, 100), (8, 1_000), (12, 100), (12, 1_000)];
        let seeds: Vec<u64> = (0..2).collect();
        let lookup_k = 10;
        let dp_max = 100_000;
        let min_w = 10;

        let mut batch_rows = Vec::new();

        for &(n, max_val) in &configs {
            let mut reports = Vec::new();
            for &seed in &seeds {
                let values = uniform_random_set(n, max_val, seed);
                let label = format!("uniform_N{}_range{}_seed{}", n, max_val, seed);
                let report = compare(&values, min_w, lookup_k, dp_max, &label);
                reports.push(report);
            }
            let config_label = format!("N={} range={}", n, max_val);
            batch_rows.push(aggregate_reports(&reports, &config_label));
        }

        print_batch_summary(&batch_rows);

        for row in &batch_rows {
            if row.avg_sas_median_err.is_finite() {
                eprintln!(
                    "{}: sas_err={:.1}%, spearman={:.3}",
                    row.config,
                    row.avg_sas_median_err * 100.0,
                    row.avg_sas_spearman
                );
            }
        }
    }

    #[test]
    fn test_empirical_distribution_comparison() {
        let Some(dist) = load_cja_or_skip("test_empirical_distribution_comparison") else {
            return;
        };
        let ns: Vec<usize> = vec![8, 12];
        let seeds: Vec<u64> = (0..2).collect();
        let lookup_k = 10;
        let dp_max = 100_000;
        let min_w = 5;

        let mut batch_rows = Vec::new();

        for &n in &ns {
            let mut reports = Vec::new();
            for &seed in &seeds {
                let values = dist.random_set(n, seed);
                let label = format!("empirical_N{}_seed{}", n, seed);
                let report = compare(&values, min_w, lookup_k, dp_max, &label);
                reports.push(report);
            }
            let config_label = format!("empirical N={}", n);
            batch_rows.push(aggregate_reports(&reports, &config_label));
        }

        print_batch_summary(&batch_rows);
    }

    #[test]
    fn test_empirical_coinjoin_comparison() {
        let Some(dist) = load_cja_or_skip("test_empirical_coinjoin_comparison") else {
            return;
        };
        let tx = random_coinjoin(&dist, 2, 2, 42);
        let report = compare(&tx.inputs, 2, 4, 100_000, "coinjoin_2p2i_seed42");
        assert!(!report.rows.is_empty() || report.n <= 4);
    }
}
