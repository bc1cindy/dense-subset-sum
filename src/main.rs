/// Given a set A = {a1, ..., aN} and a target sum E, the subset sum problem
/// asks: how many subsets of A sum to exactly E? This count is called W(E).
/// 
/// (https://arxiv.org/pdf/cond-mat/0106125)
///
/// Computing W(E) exactly requires enumerating all 2^N subsets — infeasible
/// for large N. This crate uses the asymptotic formula (3.9) from:
///
///   Sasamoto, Toyoizumi, Nishimori
///   "Statistical Mechanics of an NP-complete Problem: Subset Sum"
///
/// The formula gives W(E) ≈ numerator / denominator where:
///
///   numerator   = exp[ Σ log(1 + e^{-βaⱼ}) + β·⟨E⟩ ]
///   denominator = √(2π · Var(E))
///   ⟨E⟩         = Σ aⱼ / (1 + e^{βaⱼ})           — eq (3.2)
///   Var(E)      = Σ aⱼ² / ((1+e^{βaⱼ})(1+e^{-βaⱼ}))  — eq (3.3)
///
/// The parameter β (inverse temperature) is not given directly — it's
/// determined by the constraint E = ⟨E⟩(β) (eq 3.10). So to compute W(E)
/// for a specific E, we first invert eq (3.10) via bisection to find β,
/// then plug β into eq (3.9).
///
/// The formula is asymptotically exact: the ratio W_approx/W_exact → 1
/// as N → ∞. It should NOT be used when W(E) < 1 (the paper warns about
/// this explicitly at the end of section 3).
///
/// The practical application is identifying "dense" target values — values
/// of E where many solutions exist (W(E) >> 1), which makes the subset sum
/// instance easy to solve. A safety margin (e.g. W(E) >= 1000) ensures
/// the chosen E is well within the dense region.

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════
// CORE: formula (3.9) from Sasamoto, Toyoizumi, Nishimori
// ═══════════════════════════════════════════════════════════════

/// Eq (3.2): average energy ⟨E⟩ = Σ aⱼ / (1 + e^{βaⱼ})
///
/// As β goes from -∞ to +∞, ⟨E⟩ decreases from Σaⱼ to 0.
/// This monotonicity is what makes bisection work in find_beta.
fn mean_energy(a: &[f64], beta: f64) -> f64 {
    a.iter().map(|&aj| aj / (1.0 + (beta * aj).exp())).sum()
}

/// Eq (3.3): energy variance Var(E) = Σ aⱼ² / ((1+e^{βaⱼ})(1+e^{-βaⱼ}))
///
/// This appears in the denominator of (3.9) as √(2π·Var).
/// It comes from the second derivative of log Z (the partition function).
fn variance_energy(a: &[f64], beta: f64) -> f64 {
    a.iter().map(|&aj| {
        let x = beta * aj;
        aj * aj / ((1.0 + x.exp()) * (1.0 + (-x).exp()))
    }).sum()
}

/// Eq (3.9): log W(E) given β.
///
/// Computes log of the numerator minus log of the denominator:
///   log W = [Σ log(1 + e^{-βaⱼ}) + β·⟨E⟩] - ½·log(2π·Var)
///
/// The sum Σ log(1 + e^{-βaⱼ}) needs care for large |βaⱼ|:
///   βaⱼ > 30:  log(1 + e^{-x}) ≈ e^{-x}     (tiny correction to 0)
///   βaⱼ < -30: log(1 + e^{-x}) ≈ -x = |βaⱼ|  (the e^{|x|} term dominates)
///   otherwise:  compute directly
///
/// Working in log-space avoids overflow for any N.
fn log_w(a: &[f64], beta: f64) -> f64 {
    let sum_log: f64 = a.iter().map(|&aj| {
        let x = beta * aj;
        if      x >  30.0 { (-x).exp()           }
        else if x < -30.0 { -x                   }
        else               { (1.0 + (-x).exp()).ln() }
    }).sum();

    let e_mean  = mean_energy(a, beta);
    let var_e   = variance_energy(a, beta);
    let log_num = sum_log + beta * e_mean;
    let log_den = 0.5 * (2.0 * PI * var_e).ln();
    log_num - log_den
}

/// Invert eq (3.10): find β such that ⟨E⟩(β) = e_target.
///
/// Formula (3.9) gives W as a function of β, not E. To get W(E) for a
/// specific E, we need the β that makes ⟨E⟩(β) = E. Since ⟨E⟩ is strictly
/// decreasing in β, bisection on [-200, 200] always converges.
///
/// Returns None if e_target is outside (0, Σaⱼ).
fn find_beta(a: &[f64], e_target: f64, tol: f64) -> Option<f64> {
    let e_max: f64 = a.iter().sum();
    if e_target <= 0.0 || e_target >= e_max { return None; }

    let mut lo = -200.0_f64;
    let mut hi =  200.0_f64;
    let f = |b: f64| mean_energy(a, b) - e_target;
    if f(lo) < 0.0 || f(hi) > 0.0 { return None; }

    for _ in 0..300 {
        let mid = (lo + hi) / 2.0;
        if (hi - lo) < tol { return Some(mid); }
        if f(mid) > 0.0 { lo = mid; } else { hi = mid; }
    }
    Some((lo + hi) / 2.0)
}

/// Compute log W(E) for a given target energy E.
///
/// This is the main entry point: E → find β via (3.10) → compute log W via (3.9).
pub fn log_w_for_e(a: &[f64], e_target: f64) -> Option<f64> {
    let beta = find_beta(a, e_target, 1e-12)?;
    Some(log_w(a, beta))
}

// ═══════════════════════════════════════════════════════════════
// DENSITY: find regions of E where W(E) is large
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct DenseRegion {
    /// E value where W(E) is maximized
    pub e_peak: f64,
    /// log W at the peak
    pub log_w_peak: f64,
    /// W at the peak (Infinity if too large for f64)
    pub w_peak: f64,
    /// Lower bound of the dense interval [e_lo, e_hi]
    pub e_lo: f64,
    /// Upper bound of the dense interval
    pub e_hi: f64,
    /// Center of the dense interval — a conservative choice for E
    pub e_safe: f64,
    /// log W at the safe point
    pub log_w_safe: f64,
}

/// Scan E in `n_steps` uniform points across (0, Σaⱼ).
/// Returns (E, log W) pairs where log W >= min_log_w, sorted by W descending.
fn scan(a: &[f64], min_log_w: f64, n_steps: usize) -> Vec<(f64, f64)> {
    let e_max: f64 = a.iter().sum();
    let mut pts: Vec<(f64, f64)> = (1..n_steps)
        .filter_map(|i| {
            let e = e_max * i as f64 / n_steps as f64;
            let lw = log_w_for_e(a, e)?;
            if lw >= min_log_w { Some((e, lw)) } else { None }
        })
        .collect();
    pts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    pts
}

/// Find the densest region where log W(E) >= min_log_w.
///
/// min_log_w is the safety margin in log-space:
///   W >= 10   → min_log_w = ln(10)  ≈ 2.3
///   W >= 1000 → min_log_w = ln(1000) ≈ 6.9
///
/// Returns the peak, the valid interval, and the safe center point.
/// The safe point (center of the interval) is a conservative choice —
/// it maximizes the distance from the edges where W drops below threshold.
pub fn find_dense_region(
    a: &[f64],
    min_log_w: f64,
    n_steps: usize,
) -> Option<DenseRegion> {
    let pts = scan(a, min_log_w, n_steps);
    let &(e_peak, log_w_peak) = pts.first()?;

    let (mut e_lo, mut e_hi) = (f64::MAX, f64::MIN);
    for &(e, _) in &pts {
        if e < e_lo { e_lo = e; }
        if e > e_hi { e_hi = e; }
    }

    let e_safe = (e_lo + e_hi) / 2.0;
    let log_w_safe = log_w_for_e(a, e_safe)?;

    Some(DenseRegion {
        e_peak,
        log_w_peak,
        w_peak: if log_w_peak < 700.0 { log_w_peak.exp() } else { f64::INFINITY },
        e_lo,
        e_hi,
        e_safe,
        log_w_safe,
    })
}

/// Top-k dense candidates, sorted by W(E) descending.
pub fn dense_candidates(
    a: &[f64],
    min_log_w: f64,
    n_steps: usize,
    top_k: usize,
) -> Vec<(f64, f64)> {
    let mut pts = scan(a, min_log_w, n_steps);
    pts.truncate(top_k);
    pts
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

fn main() {
    // Example set from Figure 1 of the paper (N=16, L=256)
    let a: Vec<f64> = vec![
        218.0, 13.0, 227.0, 193.0, 70.0, 134.0, 89.0,
        198.0, 205.0, 147.0, 227.0, 190.0, 27.0, 239.0,
        192.0, 131.0,
    ];
    let e_max: f64 = a.iter().sum();

    println!("══════════════════════════════════════════════════════");
    println!(" dense-subset-sum  N={}, E_max={}", a.len(), e_max);
    println!("══════════════════════════════════════════════════════");

    // Scan W(E) across the full range
    println!("\n{:>8}  {:>10}  {:>12}", "E", "log W(E)", "W(E)");
    println!("{:─<35}", "");
    for i in 1..=19 {
        let e = e_max * i as f64 / 20.0;
        if let Some(lw) = log_w_for_e(&a, e) {
            let w_str = if lw < 700.0 { format!("{:.2}", lw.exp()) } else { "∞".into() };
            println!("{:>8.0}  {:>10.4}  {:>12}", e, lw, w_str);
        }
    }

    // Find dense region with safety margin W >= 10
    let min_log_w = 10_f64.ln();
    if let Some(r) = find_dense_region(&a, min_log_w, 1000) {
        println!("\n── Dense region (W >= 10) ───────────────────────────");
        println!("  Peak: E={:.1}, W~={:.1}", r.e_peak, r.w_peak);
        println!("  Valid range: [{:.1}, {:.1}]", r.e_lo, r.e_hi);
        println!("  Safe E: {:.0}, W~={:.1}", r.e_safe, r.log_w_safe.exp());
    }

    // Large N demo
    println!("\n── N=500, W >= 1000 ────────────────────────────────");
    let big: Vec<f64> = (1..=500).map(|i| i as f64).collect();
    if let Some(r) = find_dense_region(&big, 1000_f64.ln(), 500) {
        println!("  Safe E: {:.1}, ~= 2^{:.1} solutions",
                 r.e_safe, r.log_w_safe / 2_f64.ln());
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytical proof: when all aⱼ = 1, W(E) = C(N, E) exactly.
    /// The paper confirms this in section 6 (β = ln(N/E - 1)).
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
            assert!(err < 0.15, "C(20,{})={}, got {:.1}, err={:.1}%", k, exact, approx, err * 100.0);
        }
    }

    /// Numerical proof: enumerate all 2^20 subsets, compare with formula.
    /// Uses non-uniform elements (aⱼ = j) to test the general case.
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
            if w < 100 { continue; }
            let e_f = e as f64;
            if e_f <= 0.0 || e_f >= e_max { continue; }
            if let Some(lw) = log_w_for_e(&a, e_f) {
                let err = (lw.exp() - w as f64).abs() / w as f64;
                assert!(err < 0.20, "E={}: exact={}, approx={:.1}, err={:.1}%", e, w, lw.exp(), err * 100.0);
                tested += 1;
            }
        }
        assert!(tested > 20);
    }

    /// Structural proof: W(E) = W(E_max - E).
    /// Replacing nⱼ with 1-nⱼ maps sum S to E_max - S, so the count is symmetric.
    #[test]
    fn test_symmetry() {
        let a = vec![3.0, 7.0, 11.0, 5.0, 9.0];
        let emax: f64 = a.iter().sum();
        let l1 = log_w_for_e(&a, 15.0).unwrap();
        let l2 = log_w_for_e(&a, emax - 15.0).unwrap();
        assert!((l1 - l2).abs() < 1e-8);
    }

    /// Robustness proof: N=1000 produces finite results (no overflow).
    #[test]
    fn test_no_overflow() {
        let a: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let lw = log_w_for_e(&a, a.iter().sum::<f64>() / 2.0).unwrap();
        assert!(lw.is_finite() && lw > 100.0);
    }

    /// Convergence proof: error at peak shrinks as N grows (N=16,18,20).
    /// Paper: "the ratio of right and left hand sides tends to unity as N → ∞."
    #[test]
    fn test_convergence_with_n() {
        fn peak_error(n: usize) -> f64 {
            let a: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let e_mid = a.iter().sum::<f64>() / 2.0;
            let mut w_exact = 0u64;
            for mask in 0..(1u64 << n) {
                let sum: f64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
                if (sum - e_mid.round()).abs() < 0.5 { w_exact += 1; }
            }
            let approx = log_w_for_e(&a, e_mid.round()).unwrap().exp();
            (approx - w_exact as f64).abs() / w_exact as f64
        }

        let err_16 = peak_error(16);
        let err_18 = peak_error(18);
        let err_20 = peak_error(20);

        assert!(err_18 < err_16, "err should decrease: 16={:.4} 18={:.4}", err_16, err_18);
        assert!(err_20 < err_18, "err should decrease: 18={:.4} 20={:.4}", err_18, err_20);
        assert!(err_20 < 0.05, "N=20 err={:.1}%, want <5%", err_20 * 100.0);
    }
}
