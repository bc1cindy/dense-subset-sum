// Random-subset density crossing: "do random subsets give rise to dense subset sum?" — per-subset κ vs κ_c.

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::stats::median;
use crate::{kappa, kappa_c};

#[derive(Debug, Clone)]
pub struct SubsetDensityPoint {
    pub size: usize,
    pub n_samples: usize,
    pub fraction_dense: f64,
    pub mean_kappa: f64,
    pub mean_kappa_c: f64,
    pub median_kappa: f64,
    pub median_kappa_c: f64,
    pub fraction_undefined: f64,
}

/// For each k in `sizes`, samples `n_samples` subsets and counts `κ < κ_c` at each
/// subset's own midpoint x = (Σ/2)/(N·L) — not a fixed x=0.5, which is singular.
/// Sizes < 2 or > `values.len()` are skipped.
pub fn subset_density_sweep(
    values: &[u64],
    sizes: &[usize],
    n_samples: usize,
    seed: u64,
) -> Vec<SubsetDensityPoint> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut results = Vec::new();

    for &k in sizes {
        if k < 2 || k > values.len() {
            continue;
        }

        let mut kappas: Vec<f64> = Vec::with_capacity(n_samples);
        let mut kcs: Vec<f64> = Vec::with_capacity(n_samples);
        let mut n_dense = 0usize;
        let mut n_undef = 0usize;

        for _ in 0..n_samples {
            let subset: Vec<u64> = values.choose_multiple(&mut rng, k).copied().collect();

            let kv = match kappa(&subset) {
                Some(v) => v,
                None => {
                    n_undef += 1;
                    continue;
                }
            };

            let l = *subset.iter().max().unwrap() as f64;
            let n = subset.len() as f64;
            let sum: u64 = subset.iter().sum();
            let x = (sum as f64) / (2.0 * n * l);

            let kcv = match kappa_c(x) {
                Some(v) => v,
                None => {
                    n_undef += 1;
                    continue;
                }
            };

            if kv < kcv {
                n_dense += 1;
            }
            kappas.push(kv);
            if kcv.is_finite() {
                kcs.push(kcv);
            }
        }

        let n_valid = kappas.len();
        if n_valid == 0 {
            results.push(SubsetDensityPoint {
                size: k,
                n_samples,
                fraction_dense: f64::NAN,
                mean_kappa: f64::NAN,
                mean_kappa_c: f64::NAN,
                median_kappa: f64::NAN,
                median_kappa_c: f64::NAN,
                fraction_undefined: n_undef as f64 / n_samples as f64,
            });
            continue;
        }

        let mean_kappa = kappas.iter().sum::<f64>() / n_valid as f64;
        let (mean_kc, median_kc) = if kcs.is_empty() {
            (f64::INFINITY, f64::INFINITY)
        } else {
            (kcs.iter().sum::<f64>() / kcs.len() as f64, median(&kcs))
        };

        results.push(SubsetDensityPoint {
            size: k,
            n_samples,
            fraction_dense: n_dense as f64 / n_valid as f64,
            mean_kappa,
            mean_kappa_c: mean_kc,
            median_kappa: median(&kappas),
            median_kappa_c: median_kc,
            fraction_undefined: n_undef as f64 / n_samples as f64,
        });
    }

    results
}

pub fn print_subset_density_sweep(label: &str, n_values: usize, rows: &[SubsetDensityPoint]) {
    println!(
        "\n=== subset density sweep: {} (|values|={}) ===",
        label, n_values
    );
    println!(
        "{:>6} {:>8} {:>13} {:>11} {:>11} {:>11} {:>11}",
        "k", "samples", "frac_dense", "mean_κ", "mean_κ_c", "med_κ", "med_κ_c"
    );
    println!("{:─<76}", "");
    for r in rows {
        println!(
            "{:>6} {:>8} {:>12.2}% {:>11.4} {:>11.4} {:>11.4} {:>11.4}",
            r.size,
            r.n_samples,
            r.fraction_dense * 100.0,
            r.mean_kappa,
            r.mean_kappa_c,
            r.median_kappa,
            r.median_kappa_c
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_density_sweep_monotone_on_uniform() {
        // κ = log2(max)/k on uniform [1..L]; fraction_dense must be non-decreasing in k.
        let values: Vec<u64> = (1..=80).collect();
        let sizes = vec![4, 8, 16, 32, 64];
        let rows = subset_density_sweep(&values, &sizes, 50, 42);

        assert_eq!(rows.len(), sizes.len());
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.size, sizes[i]);
            assert!(r.fraction_dense >= 0.0 && r.fraction_dense <= 1.0);
        }

        for w in rows.windows(2) {
            assert!(
                w[1].fraction_dense + 0.05 >= w[0].fraction_dense,
                "fraction_dense should not drop sharply: k={} frac={:.2} → k={} frac={:.2}",
                w[0].size,
                w[0].fraction_dense,
                w[1].size,
                w[1].fraction_dense,
            );
        }

        // k=64 on uniform [1..80]: κ ≈ 0.1 ≪ κ_c(0.5) ≈ 0.72 ⇒ mostly dense.
        let last = rows.last().unwrap();
        assert!(
            last.fraction_dense > 0.8,
            "k=64 on uniform 1..80 should be overwhelmingly dense, got {:.2}",
            last.fraction_dense,
        );
    }

    #[test]
    fn test_subset_density_sweep_deterministic() {
        let values: Vec<u64> = (1..=40).collect();
        let sizes = vec![4, 8, 16];
        let a = subset_density_sweep(&values, &sizes, 20, 7);
        let b = subset_density_sweep(&values, &sizes, 20, 7);
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert_eq!(ra.size, rb.size);
            assert_eq!(ra.fraction_dense, rb.fraction_dense);
            assert_eq!(ra.mean_kappa, rb.mean_kappa);
            assert_eq!(ra.mean_kappa_c, rb.mean_kappa_c);
        }
    }

    #[test]
    fn test_subset_density_sweep_skips_invalid_sizes() {
        let values: Vec<u64> = (1..=10).collect();
        let sizes = vec![1, 5, 100];
        let rows = subset_density_sweep(&values, &sizes, 10, 1);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].size, 5);
    }

    #[test]
    fn test_subset_density_sweep_equal_denoms_all_sparse() {
        // Equal denoms land at x=0.5 where κ_c=0 (paper eq 4.3); any κ>0 is sparse.
        let values = vec![10_000u64; 17];
        let rows = subset_density_sweep(&values, &[4, 8, 16], 30, 7);
        assert_eq!(rows.len(), 3);
        for r in &rows {
            assert_eq!(
                r.fraction_dense, 0.0,
                "equal denoms, k={}: κ_c(0.5)=0 ⇒ κ>0 always sparse, got {:.2}",
                r.size, r.fraction_dense
            );
            assert_eq!(
                r.fraction_undefined, 0.0,
                "k={}: x=0.5 must not mark samples as undefined",
                r.size
            );
            assert_eq!(
                r.median_kappa_c, 0.0,
                "k={}: median κ_c at x=0.5 must be 0",
                r.size
            );
        }
    }

    #[test]
    fn test_subset_density_sweep_sparse_small_k() {
        // k=4 subsets are {1,1,1,1} (x=0.5 ⇒ κ_c=0) or {1,1,1,OUTLIER} (κ ≫ κ_c) — all sparse.
        let values = vec![1u64, 1, 1, 1, 10_000_000_000_000];
        let rows = subset_density_sweep(&values, &[4], 200, 3);
        assert_eq!(
            rows[0].fraction_dense, 0.0,
            "k=4 with dominant outlier: all samples sparse, got {:.2}",
            rows[0].fraction_dense
        );
    }
}
