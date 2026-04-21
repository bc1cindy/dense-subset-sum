//! Density regime: κ vs κ_c from Sasamoto eq (4.3).

/// κ = log₂(max(A))/|A|. Paper uses L (ensemble range); max(A) is the proxy.
pub fn kappa(a: &[u64]) -> Option<f64> {
    if a.is_empty() {
        return None;
    }
    let l = *a.iter().max().expect("a non-empty (checked above)");
    if l == 0 {
        return None;
    }
    Some((l as f64).log2() / a.len() as f64)
}

/// Critical density from Sasamoto eq (4.3): κ_c = (1/ln2)·[∫₀¹ ln(1+e^{-αy}) dy + α·x].
/// κ < κ_c → dense. Symmetric about 1/2; κ_c(1/4)=κ_c(3/4)=1; κ_c → 0 at x → {0, 1/2, 1}.
pub fn kappa_c(x: f64) -> Option<f64> {
    if x > 0.5 && x < 1.0 {
        return kappa_c(1.0 - x);
    }
    if x <= 0.0 || x >= 1.0 {
        return None;
    }
    if x == 0.5 {
        return Some(0.0);
    }

    let alpha = find_alpha(x)?;
    let integral = trapezoidal_integral(|s| (1.0 + (-alpha * s).exp()).ln(), 0.0, 1.0, 64);
    Some((integral + alpha * x) / 2.0_f64.ln())
}

/// Returns (κ, κ_c, κ < κ_c).
pub fn density_regime(a: &[u64], e_target: u64) -> Option<(f64, f64, bool)> {
    let k = kappa(a)?;
    let l = *a.iter().max().expect("a non-empty (kappa returned Some)") as f64;
    let n = a.len() as f64;
    let x = e_target as f64 / (n * l);

    if x <= 0.0 || x >= 1.0 {
        return None;
    }

    let kc = kappa_c(x)?;
    Some((k, kc, k < kc))
}

/// Bisects α such that ∫₀¹ s/(1+e^{α·s}) ds = x. Integral is strictly decreasing in α.
fn find_alpha(x: f64) -> Option<f64> {
    let f = |alpha: f64| -> f64 {
        trapezoidal_integral(|s| s / (1.0 + (alpha * s).exp()), 0.0, 1.0, 64)
    };

    let mut lo = -100.0_f64;
    let mut hi = 100.0_f64;

    if f(lo) < x || f(hi) > x {
        return None;
    }

    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if (hi - lo) < 1e-12 {
            return Some(mid);
        }
        if f(mid) > x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}

fn trapezoidal_integral<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = (f(a) + f(b)) / 2.0;
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Paper example: N=16, L=256 → κ = 8/16 = 0.5.
    #[test]
    fn test_kappa_paper_example() {
        let a: Vec<u64> = vec![
            218, 13, 227, 193, 70, 134, 89, 198, 205, 147, 227, 190, 27, 239, 192, 131,
        ];
        let k = kappa(&a).unwrap();
        let expected = (239.0_f64).log2() / 16.0;
        assert!(
            (k - expected).abs() < 1e-10,
            "expected {}, got {}",
            expected,
            k
        );
    }

    #[test]
    fn test_kappa_decreases_with_n() {
        let k10 = kappa(&(1..=10).map(|i| i * 100).collect::<Vec<u64>>()).unwrap();
        let k20 = kappa(&(1..=20).map(|i| i * 100).collect::<Vec<u64>>()).unwrap();
        assert!(k20 < k10, "κ should decrease: k10={}, k20={}", k10, k20);
    }

    #[test]
    fn test_kappa_empty_returns_none() {
        assert!(kappa(&[]).is_none());
    }

    #[test]
    fn test_kappa_all_zeros_returns_none() {
        assert!(kappa(&[0, 0, 0, 0]).is_none());
    }

    #[test]
    fn test_kappa_c_midpoint() {
        let kc = kappa_c(0.25).unwrap();
        assert!(
            kc > 0.0 && kc.is_finite(),
            "κ_c(0.25) = {} should be positive finite",
            kc
        );
        eprintln!("κ_c(0.25) = {:.4}", kc);
    }

    #[test]
    fn test_kappa_c_symmetry() {
        let kc1 = kappa_c(0.2).unwrap();
        let kc2 = kappa_c(0.8).unwrap();
        assert!(
            (kc1 - kc2).abs() < 1e-4,
            "κ_c should be symmetric: κ_c(0.2)={}, κ_c(0.8)={}",
            kc1,
            kc2
        );
    }

    /// Peaks at x=1/4 (α=0, κ_c=1), NOT at x=1/2 (paper Fig. 2 double-hump).
    #[test]
    fn test_kappa_c_peak_at_quartile() {
        let kc_quarter = kappa_c(0.25).unwrap();
        let kc_edge = kappa_c(0.05).unwrap();
        let kc_near_half = kappa_c(0.45).unwrap();
        assert!(
            (kc_quarter - 1.0).abs() < 1e-3,
            "κ_c(1/4) should be ≈ 1 (α = 0), got {}",
            kc_quarter,
        );
        assert!(
            kc_quarter > kc_edge,
            "κ_c(1/4) > κ_c(0.05): got {} vs {}",
            kc_quarter,
            kc_edge,
        );
        assert!(
            kc_quarter > kc_near_half,
            "κ_c(1/4) > κ_c(0.45) (valley toward 1/2): got {} vs {}",
            kc_quarter,
            kc_near_half,
        );
    }

    /// Paper Fig. 2 / eq (4.3): κ_c(1/2) = 0 (saddle α → -∞, integrand cancels).
    #[test]
    fn test_kappa_c_at_half_is_zero() {
        let kc = kappa_c(0.5).expect("κ_c(0.5) must be defined");
        assert_eq!(kc, 0.0, "κ_c(1/2) should be 0 per paper, got {}", kc);

        let a = vec![10u64; 8];
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let (k, kc, dense) =
            density_regime(&a, e_mid).expect("density_regime must succeed at x=0.5");
        assert_eq!(kc, 0.0);
        assert!(k > 0.0);
        assert!(!dense, "equal-denom midpoint is sparse (κ > 0 = κ_c)");
    }

    #[test]
    fn test_density_regime_paper() {
        let a: Vec<u64> = (1..=16).map(|i| i * 16).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        if let Some((k, kc, is_dense)) = density_regime(&a, e_mid) {
            eprintln!("κ={:.3}, κ_c={:.3}, dense={}", k, kc, is_dense);
            assert!(
                is_dense,
                "paper-like example should be dense at midpoint: κ={}, κ_c={}",
                k, kc
            );
        }
    }

    #[test]
    fn test_density_regime_large_n() {
        let a: Vec<u64> = (1..=100).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let (k, kc, is_dense) = density_regime(&a, e_mid).unwrap();
        eprintln!("N=100: κ={:.3}, κ_c={:.3}, dense={}", k, kc, is_dense);
        assert!(is_dense, "N=100 should be deeply in dense regime");
        assert!(k < 0.1, "κ should be small for large N: {}", k);
    }
}
