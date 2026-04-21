//! Sasamoto/Toyoizumi/Nishimori asymptotic W(E) via saddle-point.
//!
//! W(E) = #{S ⊆ A : ΣS = E}, the number of subsets of A summing to E.
//! Exact evaluation is #P-complete. This module implements the O(N)
//! asymptotic formula (eq 3.9) from:
//!
//!   Sasamoto, Toyoizumi, Nishimori,
//!   "Statistical Mechanics of an NP-complete Problem: Subset Sum",
//!   arxiv:cond-mat/0106125
//!
//! Output is in log space: W(E) reaches 2^N and overflows f64 past
//! N ≈ 1024. Unsound when W(E) < 1 (paper §3); callers gate use by
//! density regime before dispatching here.

use std::f64::consts::PI;

/// Public entry (f64): target energy E → saddle β (3.10) → log W(E) (3.9).
///
/// Returns log W, not W. The paper states (3.9) in linear form, but W grows
/// like 2^N near the midpoint of Σaⱼ and overflows f64 past N ≈ 1024. Log
/// space also has far better relative precision for the large values this
/// estimator is designed to produce, which is what callers consume.
pub fn log_w_for_e(a: &[f64], e_target: f64) -> Option<f64> {
    let beta = find_beta(a, e_target, 1e-12)?;
    Some(log_w(a, beta))
}

/// Public entry (u64/satoshi): GCD-normalizes A and E, then calls (3.9).
///
/// Paper assumes gcd(A) = 1 (note after eq 3.6: otherwise extra saddles of
/// equal order appear and (3.9) misses their contribution). Dividing A and E
/// by gcd(A) restores the assumption on real inputs where it rarely holds
/// (e.g. satoshi amounts in multiples of 1000). E not a multiple of gcd(A)
/// has no solution → NEG_INFINITY. None if E ∉ (0, Σaⱼ) or A empty/all-zero.
pub fn log_w_for_e_sat(a: &[u64], e_target: u64) -> Option<f64> {
    if a.is_empty() {
        return None;
    }

    let g = gcd_slice(a);
    if g == 0 {
        return None;
    }

    if !e_target.is_multiple_of(g) {
        return Some(f64::NEG_INFINITY);
    }

    let a_norm: Vec<f64> = a.iter().map(|&v| (v / g) as f64).collect();
    let e_norm = (e_target / g) as f64;

    log_w_for_e(&a_norm, e_norm)
}

/// Sasamoto critical subset size (paper appendix A.7):
/// `N_c(A) = ½·log₂(π/2·Σaⱼ²)`.
///
/// Compared against `N`: when `N ≫ N_c`, the asymptotic formula is in its
/// dense-regime sweet spot. When `N_c/(N − N_mine)` is small the W-based
/// penalty terms are reliable; when it isn't, the asymptotic is presumed
/// wrong and those terms should be gated off.
pub fn n_c(a: &[u64]) -> f64 {
    let sum_sq: f64 = a.iter().map(|&v| (v as f64).powi(2)).sum();
    if sum_sq <= 0.0 {
        return f64::NAN;
    }
    0.5 * (PI * 0.5 * sum_sq).log2()
}

pub(crate) fn gcd_slice(vals: &[u64]) -> u64 {
    vals.iter().copied().fold(0, gcd)
}

/// Solves (3.10) for β given target E: finds the saddle around which (3.9)
/// is evaluated. Bisection works because ⟨E⟩(β) is strictly monotone (3.2).
///
/// β ∈ [-200, 200]: far wider than needed; for any realistic subset-sum input
/// |β| saturates long before. None if E ∉ (0, Σaⱼ) (no valid saddle).
fn find_beta(a: &[f64], e_target: f64, tol: f64) -> Option<f64> {
    let e_max: f64 = a.iter().sum();
    if e_target <= 0.0 || e_target >= e_max {
        return None;
    }

    let mut lo = -200.0_f64;
    let mut hi = 200.0_f64;
    let f = |b: f64| mean_energy(a, b) - e_target;
    if f(lo) < 0.0 || f(hi) > 0.0 {
        return None;
    }

    for _ in 0..300 {
        let mid = (lo + hi) / 2.0;
        if (hi - lo) < tol {
            return Some(mid);
        }
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}

/// Eq (3.9): log W(E) = Σ log(1+e^{-βaⱼ}) + β⟨E⟩ − ½ log(2π·Var).
///
/// The |βaⱼ| > 30 branches are numerically-stable approximations of log(1+e^{-x}):
/// e^{-x} for x ≫ 0 and -x for x ≪ 0. Without them, `exp(±βaⱼ)` overflows
/// and the analytic limit is lost to NaN.
fn log_w(a: &[f64], beta: f64) -> f64 {
    let sum_log: f64 = a
        .iter()
        .map(|&aj| {
            let x = beta * aj;
            if x > 30.0 {
                (-x).exp()
            } else if x < -30.0 {
                -x
            } else {
                (1.0 + (-x).exp()).ln()
            }
        })
        .sum();

    let e_mean = mean_energy(a, beta);
    let var_e = variance_energy(a, beta);
    let log_num = sum_log + beta * e_mean;
    let log_den = 0.5 * (2.0 * PI * var_e).ln();
    log_num - log_den
}

/// Eq (3.2): ⟨E⟩(β) = Σ aⱼ/(1+e^{βaⱼ}). Strictly decreasing in β, so (3.10)
/// is invertible by bisection (see `find_beta`).
fn mean_energy(a: &[f64], beta: f64) -> f64 {
    a.iter().map(|&aj| aj / (1.0 + (beta * aj).exp())).sum()
}

/// Eq (3.3): Var_β(E) = ∂²log Z/∂β². Appears as √(2π·Var) in the denominator
/// of (3.9) the Gaussian width of the saddle.
fn variance_energy(a: &[f64], beta: f64) -> f64 {
    a.iter()
        .map(|&aj| {
            let x = beta * aj;
            aj * aj / ((1.0 + x.exp()) * (1.0 + (-x).exp()))
        })
        .sum()
}

fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// W(E) = C(N, E) when all aⱼ = 1 (paper §6, β = ln(N/E - 1)).
    #[test]
    fn test_binomial() {
        let a = vec![1.0; 20];
        fn binom(n: u64, k: u64) -> u64 {
            (0..k).fold(1u64, |r, i| r * (n - i) / (i + 1))
        }
        for k in 1..20 {
            let exact = binom(20, k) as f64;
            let approx = log_w_for_e(&a, k as f64).unwrap().exp();
            let err = (approx - exact).abs() / exact;
            assert!(
                err < 0.15,
                "C(20,{})={}, got {:.1}, err={:.1}%",
                k,
                exact,
                approx,
                err * 100.0
            );
        }
    }

    #[test]
    fn test_brute_force() {
        let a: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let n = a.len();
        let e_max: f64 = a.iter().sum();

        let mut w_exact = std::collections::HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: f64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry(sum.round() as u64).or_insert(0u64) += 1;
        }
        assert_eq!(w_exact.values().sum::<u64>(), 1 << n);

        let mut tested = 0;
        for (&e, &w) in &w_exact {
            if w < 100 {
                continue;
            }
            let e_f = e as f64;
            if e_f <= 0.0 || e_f >= e_max {
                continue;
            }
            if let Some(lw) = log_w_for_e(&a, e_f) {
                let err = (lw.exp() - w as f64).abs() / w as f64;
                assert!(
                    err < 0.20,
                    "E={}: exact={}, approx={:.1}, err={:.1}%",
                    e,
                    w,
                    lw.exp(),
                    err * 100.0
                );
                tested += 1;
            }
        }
        assert!(tested > 20);
    }

    #[test]
    fn test_symmetry() {
        let a = vec![3.0, 7.0, 11.0, 5.0, 9.0];
        let emax: f64 = a.iter().sum();
        let l1 = log_w_for_e(&a, 15.0).unwrap();
        let l2 = log_w_for_e(&a, emax - 15.0).unwrap();
        assert!((l1 - l2).abs() < 1e-8);
    }

    #[test]
    fn test_no_overflow() {
        let a: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let lw = log_w_for_e(&a, a.iter().sum::<f64>() / 2.0).unwrap();
        assert!(lw.is_finite() && lw > 100.0);
    }

    /// "ratio of right and left hand sides tends to unity as N → ∞."
    #[test]
    fn test_convergence_with_n() {
        fn peak_error(n: usize) -> f64 {
            let a: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let e_mid = a.iter().sum::<f64>() / 2.0;
            let mut w_exact = 0u64;
            for mask in 0..(1u64 << n) {
                let sum: f64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
                if (sum - e_mid.round()).abs() < 0.5 {
                    w_exact += 1;
                }
            }
            let approx = log_w_for_e(&a, e_mid.round()).unwrap().exp();
            (approx - w_exact as f64).abs() / w_exact as f64
        }

        let err_16 = peak_error(16);
        let err_18 = peak_error(18);
        let err_20 = peak_error(20);

        assert!(
            err_18 < err_16,
            "err should decrease: 16={:.4} 18={:.4}",
            err_16,
            err_18
        );
        assert!(
            err_20 < err_18,
            "err should decrease: 18={:.4} 20={:.4}",
            err_18,
            err_20
        );
        assert!(err_20 < 0.05, "N=20 err={:.1}%, want <5%", err_20 * 100.0);
    }

    #[test]
    fn test_gcd_normalization() {
        let a_base: Vec<u64> = vec![3, 7, 11, 5, 9];
        let a_scaled: Vec<u64> = a_base.iter().map(|&v| v * 100).collect();
        let e_base: u64 = 15;
        let e_scaled: u64 = 1500;

        let lw_base = log_w_for_e_sat(&a_base, e_base).unwrap();
        let lw_scaled = log_w_for_e_sat(&a_scaled, e_scaled).unwrap();
        assert!(
            (lw_base - lw_scaled).abs() < 1e-10,
            "base={} scaled={}",
            lw_base,
            lw_scaled
        );
    }

    #[test]
    fn test_gcd_indivisible() {
        let a: Vec<u64> = vec![10, 20, 30, 40];
        let lw = log_w_for_e_sat(&a, 15).unwrap();
        assert!(lw == f64::NEG_INFINITY, "expected -inf, got {}", lw);
    }

    #[test]
    fn test_u64_vs_brute_force() {
        let a: Vec<u64> = (1..=20).collect();
        let n = a.len();
        let e_max: u64 = a.iter().sum();

        let mut w_exact = std::collections::HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry(sum).or_insert(0u64) += 1;
        }

        let mut tested = 0;
        for (&e, &w) in &w_exact {
            if w < 100 || e == 0 || e >= e_max {
                continue;
            }
            if let Some(lw) = log_w_for_e_sat(&a, e) {
                let err = (lw.exp() - w as f64).abs() / w as f64;
                assert!(
                    err < 0.20,
                    "E={}: exact={}, approx={:.1}, err={:.1}%",
                    e,
                    w,
                    lw.exp(),
                    err * 100.0
                );
                tested += 1;
            }
        }
        assert!(tested > 20);
    }

    #[test]
    fn test_log_w_for_e_sat_empty_returns_none() {
        assert!(log_w_for_e_sat(&[], 10).is_none());
    }

    #[test]
    fn test_log_w_for_e_sat_all_zeros_returns_none() {
        assert!(log_w_for_e_sat(&[0, 0, 0], 0).is_none());
    }

    #[test]
    fn test_n_c_closed_form_all_ones() {
        let a = vec![1u64; 10];
        let expected = 0.5 * (PI * 0.5 * 10.0).log2();
        assert!((n_c(&a) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_n_c_grows_with_magnitude() {
        // Scaling A by c multiplies Σa² by c², adds log₂(c) to N_c.
        let a = vec![3u64, 7, 11, 13, 17];
        let scaled: Vec<u64> = a.iter().map(|&v| v * 100).collect();
        assert!(((n_c(&scaled) - n_c(&a)) - 100f64.log2()).abs() < 1e-9);
    }

    #[test]
    fn test_n_c_degenerate() {
        assert!(n_c(&[]).is_nan());
        assert!(n_c(&[0, 0, 0]).is_nan());
    }
}
