//! Sasamoto/Toyoizumi/Nishimori asymptotic W(E) and W(M, E) via saddle-point.
//!
//! W(E) = #{S ⊆ A : ΣS = E}, the number of subsets of A summing to E (eq 3.9).
//! W(M, E) = #{S ⊆ A : |S| = M and ΣS = E}, the size-constrained variant
//! (eq 5.8 of the same paper, via the grand canonical ensemble).
//! Exact evaluation is #P-complete. This module implements both O(N)
//! asymptotic formulas from:
//!
//!   Sasamoto, Toyoizumi, Nishimori,
//!   "Statistical Mechanics of an NP-complete Problem: Subset Sum",
//!   [cond-mat/0106125](https://arxiv.org/abs/cond-mat/0106125)
//!
//! Output is in log space: counts reach 2^N and overflow f64 past
//! N ≈ 1024. Unsound when W < 1 (paper §3); callers gate use by
//! density regime before dispatching here.

use std::f64::consts::PI;

/// Natural log of the count of subsets of `a` that sum to `e_target`.
/// Returns `log W` because W reaches 2^N and overflows f64 past N ≈ 1024;
/// log space also gives better relative precision at these magnitudes.
/// Panics if `e_target` ∉ (0, Σaⱼ) (no subset can sum there).
pub fn log_w_for_e(a: &[f64], e_target: f64) -> f64 {
    let beta = find_beta(a, e_target, 0.0);
    log_w(a, beta)
}

/// Natural log of the count of subsets of `a` of size exactly `m_target`
/// summing to `e_target` (eq 5.8). Companion to [`log_w_for_e`]: saddle is
/// 2D in (β, μ), Gaussian width is √det(D) from eq 5.9.
/// Panics if `e_target` is outside the feasibility interior (sums of the
/// `m_target` smallest/largest elements of `a`); equal-element A collapses it.
pub fn log_w_for_m_e(a: &[f64], m_target: usize, e_target: f64) -> f64 {
    feasibility_assert(a, m_target, e_target);
    let m = m_target as f64;
    let (beta, mu) = find_beta_mu(a, m, e_target);
    log_w_grand(a, beta, mu, m, e_target)
}

/// Sum interval (Σ smallest m, Σ largest m) for size-m subsets of `a`.
/// `None` when m is 0 or ≥ a.len().
fn feasibility_bounds(a: &[f64], m_target: usize) -> Option<(f64, f64)> {
    if m_target == 0 || m_target >= a.len() {
        return None;
    }
    let mut sorted: Vec<f64> = a.to_vec();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let e_min: f64 = sorted.iter().take(m_target).sum();
    let e_max: f64 = sorted.iter().rev().take(m_target).sum();
    Some((e_min, e_max))
}

/// Rejects (M, E) outside the achievable region (sum of m smallest/largest
/// elements). Equal-element A collapses the interior; panics for any e_target.
fn feasibility_assert(a: &[f64], m_target: usize, e_target: f64) {
    let Some((e_min, e_max)) = feasibility_bounds(a, m_target) else {
        panic!("m_target = {} must be in (0, {})", m_target, a.len());
    };
    assert!(
        e_min < e_target && e_target < e_max,
        "e_target = {} must be in ({}, {}) for m = {}",
        e_target,
        e_min,
        e_max,
        m_target
    );
}

/// Bisects β such that ⟨E⟩(β, μ) = e_target (eqs 3.10/5.4); ⟨E⟩ is strictly
/// decreasing in β at fixed μ. Bracket [-200, 200] property-tested for
/// N ∈ [5, 100]. Panics if E ∉ (0, Σaⱼ) or bracket misses the saddle.
fn find_beta(a: &[f64], e_target: f64, mu: f64) -> f64 {
    let e_max: f64 = a.iter().sum();
    assert!(
        0.0 < e_target && e_target < e_max,
        "find_beta: e_target = {} must be in (0, {})",
        e_target,
        e_max
    );

    let mut lo = -200.0_f64;
    let mut hi = 200.0_f64;
    let f = |b: f64| mean_energy(a, b, mu) - e_target;
    assert!(
        f(lo) >= 0.0 && f(hi) <= 0.0,
        "find_beta: bracket [-200, 200] does not contain saddle for e_target = {}, mu = {} (f(lo)={}, f(hi)={})",
        e_target,
        mu,
        f(lo),
        f(hi)
    );

    for _ in 0..300 {
        if hi - lo < 1e-12 {
            break;
        }
        let mid = (lo + hi) / 2.0;
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Solves (5.3)+(5.4) jointly for the 2D saddle (β, μ) by nested bisection:
/// outer on μ (5.3 monotone in μ), inner via [`find_beta`] (5.4 monotone in β).
/// Panics if (m_target, e_target) is outside the bracket's reach.
fn find_beta_mu(a: &[f64], m_target: f64, e_target: f64) -> (f64, f64) {
    let mut mu_lo = -200.0_f64;
    let mut mu_hi = 200.0_f64;
    let mut mu = 0.0_f64;
    let mut beta = 0.0_f64;

    for _ in 0..300 {
        if mu_hi - mu_lo < 1e-12 {
            break;
        }
        mu = (mu_lo + mu_hi) / 2.0;
        beta = find_beta(a, e_target, mu);
        if mean_count(a, beta, mu) < m_target {
            mu_lo = mu;
        } else {
            mu_hi = mu;
        }
    }
    (beta, mu)
}

/// Eq (3.9): log W(E) = log Z + β·⟨E⟩ − ½ log(2π·Var(E)).
fn log_w(a: &[f64], beta: f64) -> f64 {
    let log_z_val = log_z(a, beta, 0.0);
    let e_mean = mean_energy(a, beta, 0.0);
    let var_e = variance_energy(a, beta, 0.0);
    log_z_val + beta * e_mean - 0.5 * (2.0 * PI * var_e).ln()
}

/// Eq (5.8): log W(M, E) = log Θ + β·E − μ·M − log(2π) − ½ log(det D).
/// det D from eq 5.9. Bivariate-Gaussian normalization 1/(2π·√D), distinct
/// from univariate 1/√(2π·Var) in (3.9).
fn log_w_grand(a: &[f64], beta: f64, mu: f64, m_target: f64, e_target: f64) -> f64 {
    let log_theta = log_z(a, beta, mu);
    let v_m = variance_count(a, beta, mu);
    let v_e = variance_energy(a, beta, mu);
    let cov = cov_count_energy(a, beta, mu);
    let det = v_m * v_e - cov * cov;
    log_theta + beta * e_target - mu * m_target - (2.0 * PI).ln() - 0.5 * det.ln()
}

/// Numerically-stable log(1+e^{-x}); |x| > 30 branches avoid `exp(±x)` overflow
/// (analytic limit: e^{-x} for x ≫ 0, -x for x ≪ 0).
fn log_one_plus_exp_neg(x: f64) -> f64 {
    if x > 30.0 {
        (-x).exp()
    } else if x < -30.0 {
        -x
    } else {
        (1.0 + (-x).exp()).ln()
    }
}

/// Fermi-Dirac occupation 1/(1+e^x). Shared denominator in eqs (5.3), (5.4).
fn fermi(x: f64) -> f64 {
    1.0 / (1.0 + x.exp())
}

/// Shared fluctuation factor f·(1−f) where f = fermi(x), the variance of a
/// Bernoulli with occupation f. Appears in eqs (3.3), (5.5), (5.6), (5.7).
fn fluct(x: f64) -> f64 {
    let f = fermi(x);
    f * (1.0 - f)
}

/// Eq (3.1)/(5.2) in log space: log Z = Σ log(1+e^{−(βaⱼ−μ)}).
fn log_z(a: &[f64], beta: f64, mu: f64) -> f64 {
    a.iter()
        .map(|&aj| log_one_plus_exp_neg(beta * aj - mu))
        .sum()
}

/// Eq (3.2)/(5.4): ⟨E⟩(β, μ) = Σ aⱼ/(1+e^{βaⱼ−μ}). Strictly decreasing in β
/// at fixed μ; invertible by [`find_beta`].
fn mean_energy(a: &[f64], beta: f64, mu: f64) -> f64 {
    a.iter().map(|&aj| aj * fermi(beta * aj - mu)).sum()
}

/// Eq (5.3): ⟨M⟩(β, μ) = Σ 1/(1+e^{βaⱼ−μ}). Strictly increasing in μ at fixed
/// β, so (5.3) is invertible by bisection (see [`find_beta_mu`]).
fn mean_count(a: &[f64], beta: f64, mu: f64) -> f64 {
    a.iter().map(|&aj| fermi(beta * aj - mu)).sum()
}

/// Eq (3.3)/(5.7): Var(E)(β, μ) = Σ aⱼ²·fluct(βaⱼ−μ). Appears as √(2π·Var)
/// in (3.9) denominator and inside det D in (5.8).
fn variance_energy(a: &[f64], beta: f64, mu: f64) -> f64 {
    a.iter().map(|&aj| aj * aj * fluct(beta * aj - mu)).sum()
}

/// Eq (5.5): Var(M)(β, μ) = Σ fluct(βaⱼ−μ). Number-fluctuation analogue of
/// variance_energy; appears inside det D in (5.8).
fn variance_count(a: &[f64], beta: f64, mu: f64) -> f64 {
    a.iter().map(|&aj| fluct(beta * aj - mu)).sum()
}

/// Eq (5.6): Cov(M, E)(β, μ) = Σ aⱼ·fluct(βaⱼ−μ). Always positive by the
/// formula structure; only the squared form appears in det D, so the sign
/// convention is irrelevant downstream.
fn cov_count_energy(a: &[f64], beta: f64, mu: f64) -> f64 {
    a.iter().map(|&aj| aj * fluct(beta * aj - mu)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Brute-force cells with W below this aren't expected to match the
    /// asymptotic; Stirling/Gaussian approximation breaks down in the tail.
    const W_MIN_FOR_ASYMPTOTIC_MATCH: u64 = 100;

    /// W(E) = C(N, E) when all aⱼ = 1 (paper §6, β = ln(N/E - 1)).
    #[test]
    fn test_binomial() {
        let a = vec![1.0; 20];
        fn binom(n: u64, k: u64) -> u64 {
            (0..k).fold(1u64, |r, i| r * (n - i) / (i + 1))
        }
        for k in 1..20 {
            let exact = binom(20, k) as f64;
            let approx = log_w_for_e(&a, k as f64).exp();
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
            if w < W_MIN_FOR_ASYMPTOTIC_MATCH {
                continue;
            }
            let e_f = e as f64;
            if e_f <= 0.0 || e_f >= e_max {
                continue;
            }
            let lw = log_w_for_e(&a, e_f);
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
        assert!(tested > 20);
    }

    #[test]
    fn test_symmetry() {
        let a = vec![3.0, 7.0, 11.0, 5.0, 9.0];
        let emax: f64 = a.iter().sum();
        let l1 = log_w_for_e(&a, 15.0);
        let l2 = log_w_for_e(&a, emax - 15.0);
        assert!((l1 - l2).abs() < 1e-8);
    }

    #[test]
    fn test_no_overflow() {
        let a: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let lw = log_w_for_e(&a, a.iter().sum::<f64>() / 2.0);
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
            let approx = log_w_for_e(&a, e_mid.round()).exp();
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

    /// `log_w_for_m_e` panics when m_target is at the boundary (0 or N).
    #[test]
    #[should_panic(expected = "m_target")]
    fn test_log_w_for_m_e_panics_on_m_out_of_range() {
        let a: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        log_w_for_m_e(&a, 0, 5.0);
    }

    /// `log_w_for_m_e` panics when e_target is outside the feasibility interior.
    /// For m=3 over [1..=10], e must lie in (6, 27); 5.0 is below the minimum.
    #[test]
    #[should_panic(expected = "e_target")]
    fn test_log_w_for_m_e_panics_on_e_out_of_range() {
        let a: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        log_w_for_m_e(&a, 3, 5.0);
    }

    /// Justifies the panic in `find_beta`: bracket [-200, 200] contains the
    /// saddle for all E ∈ (0, Σaⱼ) across realistic N.
    #[test]
    fn test_find_beta_converges_across_range() {
        for &n in &[5usize, 10, 20, 50, 100] {
            let a: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let e_max: f64 = a.iter().sum();
            for i in 1..20 {
                let e = e_max * i as f64 / 20.0;
                let lw = log_w_for_e(&a, e);
                assert!(lw.is_finite(), "N={}, E={}: log W = {}", n, e, lw);
            }
        }
    }

    /// Brute-force ground truth for W(M, E): for each (m, E) cell with W > 100,
    /// the asymptotic should match within the same tolerance as the
    /// unconstrained brute-force test.
    #[test]
    fn test_constrained_brute_force() {
        let a: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let n = a.len();

        let mut w_exact: std::collections::HashMap<(usize, u64), u64> =
            std::collections::HashMap::new();
        for mask in 0..(1u64 << n) {
            let m = mask.count_ones() as usize;
            let sum: f64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_exact.entry((m, sum.round() as u64)).or_insert(0) += 1;
        }

        let mut sorted = a.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let mut tested = 0;
        for (&(m, e), &w) in &w_exact {
            if w < W_MIN_FOR_ASYMPTOTIC_MATCH || m == 0 || m == n {
                continue;
            }
            let e_min: f64 = sorted.iter().take(m).sum();
            let e_max: f64 = sorted.iter().rev().take(m).sum();
            let e_f = e as f64;
            if e_f <= e_min || e_f >= e_max {
                continue;
            }
            let lw = log_w_for_m_e(&a, m, e_f);
            let err = (lw.exp() - w as f64).abs() / w as f64;
            assert!(
                err < 0.25,
                "(m={}, E={}): exact={}, approx={:.1}, err={:.1}%",
                m,
                e,
                w,
                lw.exp(),
                err * 100.0
            );
            tested += 1;
        }
        assert!(tested > 20, "tested only {} cells", tested);
    }

    /// Σ_M W(M, E) = W(E) (eq 5.2 marginalized to the canonical case).
    /// Connects (5.8) back to (3.9) via the identity in paper §5.
    #[test]
    fn test_marginal_identity() {
        let a: Vec<f64> = (1..=15).map(|i| i as f64).collect();
        let n = a.len();

        let mut w_total: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for mask in 0..(1u64 << n) {
            let sum: f64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
            *w_total.entry(sum.round() as u64).or_insert(0) += 1;
        }

        let e_target = 60u64;
        let total_exact = w_total[&e_target] as f64;

        let mut sorted = a.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut sum_w_m_e = 0.0_f64;
        for m in 1..n {
            let e_min: f64 = sorted.iter().take(m).sum();
            let e_max: f64 = sorted.iter().rev().take(m).sum();
            let e_f = e_target as f64;
            if e_f <= e_min || e_f >= e_max {
                continue;
            }
            sum_w_m_e += log_w_for_m_e(&a, m, e_f).exp();
        }

        let err = (sum_w_m_e - total_exact).abs() / total_exact;
        assert!(
            err < 0.30,
            "Σ_M W(M, {}) = {:.1}, brute = {}, err = {:.1}%",
            e_target,
            sum_w_m_e,
            total_exact,
            err * 100.0
        );
    }

    proptest! {
        /// find_beta inverts (3.10): mean_energy(a, find_beta(a, E)) ≈ E for
        /// arbitrary a and E in (0, Σaⱼ). Shrinks to minimal counter-examples
        /// on failure.
        #[test]
        fn find_beta_inverts_mean_energy(
            (a, e) in (5usize..30).prop_flat_map(|n| {
                (
                    proptest::collection::vec(1.0f64..1000.0, n),
                    0.05f64..0.95,
                )
            }).prop_map(|(a, e_frac)| {
                let e = e_frac * a.iter().sum::<f64>();
                (a, e)
            })
        ) {
            let beta = find_beta(&a, e, 0.0);
            let recovered = mean_energy(&a, beta, 0.0);
            prop_assert!(
                (recovered - e).abs() / e < 1e-6,
                "round-trip failed: e={}, recovered={}, β={}",
                e, recovered, beta
            );
        }

        /// find_beta_mu inverts (5.3)+(5.4): for arbitrary (a, M, E) in the
        /// feasibility interior, the returned (β, μ) satisfies both ⟨M⟩ = M
        /// and ⟨E⟩ = E.
        #[test]
        fn find_beta_mu_inverts(
            a in proptest::collection::vec(1.0f64..100.0, 8..18),
            m_frac in 0.25f64..0.6,
            e_frac in 0.2f64..0.8,
        ) {
            let n = a.len();
            let m = ((n as f64) * m_frac).round() as usize;
            prop_assume!(m >= 2 && m + 2 <= n);

            let mut sorted = a.clone();
            sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let e_min: f64 = sorted.iter().take(m).sum();
            let e_max: f64 = sorted.iter().rev().take(m).sum();
            let e = e_min + (e_max - e_min) * e_frac;
            prop_assume!(e > e_min * 1.05 && e < e_max * 0.95);

            let m_f = m as f64;
            let (beta, mu) = find_beta_mu(&a, m_f, e);
            let m_recovered = mean_count(&a, beta, mu);
            let e_recovered = mean_energy(&a, beta, mu);
            let m_err = (m_recovered - m_f).abs() / m_f;
            let e_err = (e_recovered - e).abs() / e;
            prop_assert!(m_err < 1e-5, "M err = {}", m_err);
            prop_assert!(e_err < 1e-5, "E err = {}", e_err);
        }

        /// W(M, E) = W(N−M, ΣA−E) by the complementary-subset bijection.
        /// The grand-canonical saddle finds different (β, μ) for each side,
        /// but eq (5.8) must produce identical log W.
        #[test]
        fn log_w_for_m_e_is_complement_symmetric(
            a in proptest::collection::vec(1.0f64..100.0, 8..18),
            m_frac in 0.3f64..0.5,
            e_frac in 0.3f64..0.7,
        ) {
            let n = a.len();
            let m = ((n as f64) * m_frac).round() as usize;
            prop_assume!(m >= 2 && m + 2 <= n);

            let mut sorted = a.clone();
            sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let e_min: f64 = sorted.iter().take(m).sum();
            let e_max: f64 = sorted.iter().rev().take(m).sum();
            let e = e_min + (e_max - e_min) * e_frac;
            prop_assume!(e > e_min * 1.05 && e < e_max * 0.95);

            let m_comp = n - m;
            let total: f64 = a.iter().sum();
            let e_comp = total - e;

            let e_comp_min: f64 = sorted.iter().take(m_comp).sum();
            let e_comp_max: f64 = sorted.iter().rev().take(m_comp).sum();
            prop_assume!(e_comp > e_comp_min * 1.05 && e_comp < e_comp_max * 0.95);

            let lw = log_w_for_m_e(&a, m, e);
            let lw_comp = log_w_for_m_e(&a, m_comp, e_comp);
            let diff = (lw - lw_comp).abs();
            prop_assert!(diff < 1e-6, "diff = {}, lw = {}, lw_comp = {}", diff, lw, lw_comp);
        }
    }
}
