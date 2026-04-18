//! Cross-validation of estimators against brute force and CJA mappings.

mod density_sweep;
mod per_coin;
mod sub_tx_estimates;

pub use density_sweep::{SubsetDensityPoint, print_subset_density_sweep, subset_density_sweep};
pub use per_coin::{
    CoinRole, CoinScore, per_coin_scores_signed, per_coin_scores_signed_fee_aware,
    print_per_coin_scores,
};
pub use sub_tx_estimates::{SubTxEstimate, estimate_sub_txs};

pub(super) fn exclude_values(full: &[u64], to_remove: &[u64]) -> Vec<u64> {
    let mut remaining = full.to_vec();
    for &val in to_remove {
        if let Some(pos) = remaining.iter().position(|&v| v == val) {
            remaining.swap_remove(pos);
        }
    }
    remaining
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclude_values() {
        let full = vec![10, 20, 10, 30, 10];
        let remove = vec![10, 10];
        let remaining = exclude_values(&full, &remove);
        assert_eq!(remaining.len(), 3);
        assert_eq!(
            remaining.iter().filter(|&&v| v == 10).count(),
            1,
            "should remove exactly 2 of 3 tens"
        );
    }

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
