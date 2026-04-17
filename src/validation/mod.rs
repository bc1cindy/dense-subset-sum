//! Cross-validation of estimators against brute force and CJA mappings.

mod density_sweep;

pub use density_sweep::{SubsetDensityPoint, print_subset_density_sweep, subset_density_sweep};

#[cfg(test)]
mod tests {
    #[test]
    fn test_compare_estimators_n20() {
        let a: Vec<u64> = (1..=20).collect();
        let report = crate::comparison::compare(&a, 100, 10, 1_000_000, "n20");

        assert!(report.sasamoto.n_points > 0);
        assert!(
            report.sasamoto.median_error < 0.10,
            "sasamoto median err {:.1}%",
            report.sasamoto.median_error * 100.0
        );
        assert!(
            report.sasamoto.spearman > 0.95,
            "sasamoto spearman {:.3}",
            report.sasamoto.spearman
        );

        for row in &report.rows {
            if let Some(wl) = row.w_lookup {
                assert!(
                    wl as f64 <= row.w_exact * 1.01,
                    "lookup must be lower bound at E={}",
                    row.e_target
                );
            }
            if let Some(err) = row.err_dp {
                assert!(err.abs() < 1e-10, "DP must be exact at E={}", row.e_target);
            }
        }
    }

    #[test]
    fn test_compare_gcd_values() {
        let a: Vec<u64> = vec![
            10_000, 10_000, 10_000, 10_000, 10_000, 20_000, 20_000, 20_000, 50_000, 50_000, 50_000,
            50_000, 100_000, 100_000, 100_000,
        ];
        let report = crate::comparison::compare(&a, 10, 8, 10_000_000, "gcd");

        if report.sasamoto.n_points > 0 {
            eprintln!(
                "GCD test: sasamoto median_err={:.1}%, spearman={:.3}, n_points={}",
                report.sasamoto.median_error * 100.0,
                report.sasamoto.spearman,
                report.sasamoto.n_points
            );
        }

        for row in &report.rows {
            if let Some(err) = row.err_dp {
                assert!(
                    err.abs() < 1e-10,
                    "DP with GCD should be exact: E={}, error={:.2e}",
                    row.e_target,
                    err.abs()
                );
            }
        }
    }
}
