//! Tells whether Sasamoto's saddle-point approximation of W(E) is
//! reliable for a given subset-sum instance (dense regime, κ < κ_c)
//! or not (sparse, κ > κ_c). κ = log₂(L)/N; κ_c from eq (4.3).

use std::f64::consts::LN_2;

/// Bitcoin supply cap in satoshis.
pub const MAX_MONEY: u64 = 2_100_000_000_000_000;

/// κ = log₂(L)/N. Paper defines L as ensemble range; caller picks L.
/// max(A) ≤ L < MAX_MONEY. log₂(MAX_MONEY) ~ 51, and log₂(1e8) ~ 27.
pub fn kappa(log_l: f64, n: usize) -> Option<f64> {
    if n == 0 {
        return None;
    }
    Some(log_l / n as f64)
}

/// κ with L = MAX_MONEY/N, an upper bound on L independent of A (worst case).
pub fn worst_case_kappa(n: usize) -> f64 {
    let l = MAX_MONEY / n as u64;
    (l as f64).log2() / n as f64
}

/// Critical value for kappa, below which most instances are expected to be dense. Defined in Sasamoto eq (4.2, 4.3).
/// Panics if (N·L) < E.
pub fn kappa_c(e: u64, n: usize, l: u64) -> f64 {
    let x = e as f64 / (n as f64 * l as f64);
    assert!(0.0 < x && x <= 1.0, "x = {} must be in (0, 1]", x);
    kappa_c_at(x)
}

/// Dense only if κ < κ_c under the pessimistic L; Sparse only if κ ≥ κ_c under
/// the optimistic L; Transitional when the best/worst interval crosses κ_c.
pub enum Regime {
    Dense,
    Transitional,
    Sparse,
}

pub struct DensityRegime {
    pub best_case_kappa: f64,
    pub worst_case_kappa: f64,
    pub best_case_kappa_c: f64,
    pub worst_case_kappa_c: f64,
}

impl DensityRegime {
    /// Panics if `a` is empty or (N·L) < E.
    pub fn new<I>(a: I, e_target: u64) -> Self
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator,
    {
        let a = a.into_iter();
        let n = a.len();
        let l_best = a.max().expect("a must be non-empty");
        let l_worst = MAX_MONEY / n as u64;

        Self {
            best_case_kappa: (l_best as f64).log2() / n as f64,
            worst_case_kappa: worst_case_kappa(n),
            best_case_kappa_c: kappa_c(e_target, n, l_best),
            worst_case_kappa_c: kappa_c(e_target, n, l_worst),
        }
    }

    /// Conservative on both sides: Dense only when even the pessimistic L agrees,
    /// Sparse only when even the optimistic L agrees. Anything else is Transitional.
    pub fn regime(&self) -> Regime {
        if self.worst_case_kappa < self.worst_case_kappa_c {
            Regime::Dense
        } else if self.best_case_kappa >= self.best_case_kappa_c {
            Regime::Sparse
        } else {
            Regime::Transitional
        }
    }

    pub fn is_dense(&self) -> bool {
        matches!(self.regime(), Regime::Dense)
    }

    pub fn is_sparse(&self) -> bool {
        matches!(self.regime(), Regime::Sparse)
    }
}

/// κ_c as a function of x = E/(N·L). Symmetric about 1/2.
/// Domain: x ∈ [0, 1]. κ_c(1/4) = κ_c(3/4) = 1; κ_c = 0 at x ∈ {0, 1/2, 1}.
/// For x below find_alpha's numerical bracket, use the paper's asymptotic
/// expansion of eq (4.2, 4.3) in the α → ∞ limit: κ_c ≈ 2·√x / ln 2.
fn kappa_c_at(x: f64) -> f64 {
    if x > 0.5 {
        return kappa_c_at(1.0 - x);
    }
    // Eq (4.3) at x = 1/2: α → -∞, integrand cancels. Mirrors the canonical
    // breakdown for systems with decreasing W(E) discussed in §6.
    if x == 0.5 {
        return 0.0;
    }
    if x < 1e-3 {
        return 2.0 * x.sqrt() / LN_2;
    }
    let alpha = find_alpha(x);
    let integral = trapezoidal_integral(|s| (1.0 + (-alpha * s).exp()).ln(), 0.0, 1.0, 64);
    (integral + alpha * x) / LN_2
}

/// Bisects α such that ∫₀¹ s/(1+e^(α·s)) ds = x. Integral is strictly decreasing in α.
fn find_alpha(x: f64) -> f64 {
    let f = |alpha: f64| -> f64 {
        trapezoidal_integral(|s| s / (1.0 + (alpha * s).exp()), 0.0, 1.0, 64)
    };
    let mut lo = -1000.0_f64;
    let mut hi = 1000.0_f64;
    assert!(
        f(lo) >= x && f(hi) <= x,
        "find_alpha: x = {} outside bracket [f(-1000)={}, f(1000)={}]",
        x,
        f(lo),
        f(hi)
    );
    for _ in 0..200 {
        if hi - lo < 1e-12 {
            break;
        }
        let mid = (lo + hi) / 2.0;
        if f(mid) > x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
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
    use proptest::prelude::*;

    /// Paper's A (N=16). Paper parameterizes ensemble with L=256 (κ=0.5);
    /// best_case uses max(A)=239, so the expected κ is slightly lower.   
    #[test]
    fn test_best_case_kappa_paper_example() {
        let a: Vec<u64> = vec![
            218, 13, 227, 193, 70, 134, 89, 198, 205, 147, 227, 190, 27, 239, 192, 131,
        ];
        let e = a.iter().sum::<u64>() / 2;
        let k = DensityRegime::new(a.iter().copied(), e).best_case_kappa;
        let expected = (239.0_f64).log2() / 16.0;
        assert!(
            (k - expected).abs() < 1e-10,
            "expected {}, got {}",
            expected,
            k
        );
    }

    #[test]
    fn test_best_case_kappa_decreases_with_n() {
        let a10: Vec<u64> = (1..=10).map(|i| i * 100).collect();
        let a20: Vec<u64> = (1..=20).map(|i| i * 100).collect();
        let e10 = a10.iter().sum::<u64>() / 2;
        let e20 = a20.iter().sum::<u64>() / 2;
        let k10 = DensityRegime::new(a10.iter().copied(), e10).best_case_kappa;
        let k20 = DensityRegime::new(a20.iter().copied(), e20).best_case_kappa;
        assert!(k20 < k10, "κ should decrease: k10={}, k20={}", k10, k20);
    }

    #[test]
    #[should_panic(expected = "a must be non-empty")]
    fn test_density_regime_new_empty_panics() {
        DensityRegime::new(std::iter::empty::<u64>(), 1);
    }

    #[test]
    fn test_kappa_primitive_zero_n() {
        assert!(kappa(10.0, 0).is_none());
    }

    #[test]
    fn test_kappa_c_midpoint() {
        let kc = kappa_c_at(0.25);
        assert!(
            kc > 0.0 && kc.is_finite(),
            "κ_c(0.25) = {} should be positive finite",
            kc
        );
        eprintln!("κ_c(0.25) = {:.4}", kc);
    }

    #[test]
    fn test_kappa_c_symmetry() {
        let kc1 = kappa_c_at(0.2);
        let kc2 = kappa_c_at(0.8);
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
        let kc_quarter = kappa_c_at(0.25);
        let kc_edge = kappa_c_at(0.05);
        let kc_near_half = kappa_c_at(0.45);
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
        assert_eq!(kappa_c_at(0.5), 0.0);

        let a = [10u64; 8];
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let dr = DensityRegime::new(a.iter().copied(), e_mid);
        assert_eq!(dr.best_case_kappa_c, 0.0);
        assert!(dr.best_case_kappa > 0.0);
        assert!(
            !dr.is_dense(),
            "equal-denom midpoint is sparse (κ > 0 = κ_c)"
        );
    }

    #[test]
    fn test_kappa_c_e_matches_x() {
        assert!((kappa_c(100, 10, 50) - kappa_c_at(0.2)).abs() < 1e-12);
    }

    /// Equal-amount CoinJoin (e.g. Whirlpool) with no change: E = Σ A exactly,
    /// giving x = E/(N·max(A)) = 1. Must not panic; κ_c(1) = 0 per paper.
    #[test]
    fn test_kappa_c_equal_values_whirlpool() {
        let a: Vec<u64> = vec![100_000_000; 8];
        let e_total: u64 = a.iter().sum();
        let l = *a.iter().max().unwrap();

        assert_eq!(kappa_c(e_total, a.len(), l), 0.0);

        let dr = DensityRegime::new(a.iter().copied(), e_total);
        assert_eq!(dr.best_case_kappa_c, 0.0);
        assert!(!dr.is_dense(), "equal-value tx at E=Σ: κ > 0 = κ_c");
    }

    proptest! {
        /// find_alpha inverts the integral: f(find_alpha(x)) ≈ x for x in the
        /// supported numerical domain. Shrinks to minimal counter-examples on failure.
        #[test]
        fn find_alpha_inverts_integral(x in 1e-4f64..(0.5 - 1e-4)) {
            let alpha = find_alpha(x);
            let f_at_alpha = trapezoidal_integral(
                |s| s / (1.0 + (alpha * s).exp()), 0.0, 1.0, 64);
            prop_assert!(
                (f_at_alpha - x).abs() < 1e-6,
                "f(find_alpha({})) = {}, differs by {}", x, f_at_alpha, (f_at_alpha - x).abs()
            );
        }

        #[test]
        fn kappa_c_well_defined_on_realistic_inputs(
            n in 10usize..500,
            l in 1_000u64..2_100_000_000_000_000_u64,
            fraction in 0.01f64..0.99,
        ) {
            let e = (fraction * n as f64 * l as f64) as u64;
            prop_assume!(e > 0 && (e as f64) < n as f64 * l as f64);
            let kc = kappa_c(e, n, l);
            prop_assert!(kc.is_finite() && kc >= 0.0);
        }
    }

    #[test]
    fn trapezoidal_exact_on_constants() {
        assert_eq!(trapezoidal_integral(|_| 0.0, 0.0, 1.0, 64), 0.0);
        assert_eq!(trapezoidal_integral(|_| 1.0, 0.0, 1.0, 64), 1.0);
        assert!((trapezoidal_integral(|_| 3.5, 2.0, 5.0, 10) - 10.5).abs() < 1e-12);
    }

    #[test]
    fn trapezoidal_exact_on_linear() {
        // ∫₀¹ x dx = 1/2; trapezoidal is exact on linear functions.
        assert!((trapezoidal_integral(|x| x, 0.0, 1.0, 64) - 0.5).abs() < 1e-12);
        // ∫₂⁵ x dx = (25 - 4)/2 = 10.5.
        assert!((trapezoidal_integral(|x| x, 2.0, 5.0, 10) - 10.5).abs() < 1e-12);
    }

    #[test]
    fn trapezoidal_propagates_f64_specials() {
        assert!(trapezoidal_integral(|_| f64::INFINITY, 0.0, 1.0, 8).is_infinite());
        assert!(trapezoidal_integral(|_| f64::NEG_INFINITY, 0.0, 1.0, 8).is_infinite());
        assert!(trapezoidal_integral(|_| f64::NAN, 0.0, 1.0, 8).is_nan());
        // ∫₀¹ c dx = c for constant c.
        assert_eq!(
            trapezoidal_integral(|_| f64::MIN_POSITIVE, 0.0, 1.0, 8),
            f64::MIN_POSITIVE
        );
        assert_eq!(
            trapezoidal_integral(|_| f64::EPSILON, 0.0, 1.0, 8),
            f64::EPSILON
        );
    }

    #[test]
    fn trapezoidal_approximates_exp() {
        // ∫₀¹ e^x dx = e - 1. Trapezoidal error ~ O(max|f''|·(b-a)³/n²).
        let expected = std::f64::consts::E - 1.0;
        let result = trapezoidal_integral(|x| x.exp(), 0.0, 1.0, 256);
        assert!(
            (result - expected).abs() < 1e-5,
            "result={}, expected={}",
            result,
            expected
        );
    }

    #[test]
    fn trapezoidal_approximates_ln() {
        // ∫₁² ln(x) dx = 2·ln(2) - 1.
        let expected = 2.0 * 2.0_f64.ln() - 1.0;
        let result = trapezoidal_integral(|x| x.ln(), 1.0, 2.0, 256);
        assert!(
            (result - expected).abs() < 1e-6,
            "result={}, expected={}",
            result,
            expected
        );
    }

    #[test]
    fn test_find_alpha_outside_domain_panics() {
        let unsupported = [1e-100, f64::EPSILON, f64::MIN_POSITIVE, 0.5 - f64::EPSILON];
        for x in unsupported {
            let result = std::panic::catch_unwind(|| find_alpha(x));
            assert!(result.is_err(), "find_alpha({:e}) should panic", x);
        }
    }

    #[test]
    fn test_kappa_c_at_finite_across_range() {
        for i in 1..=999 {
            let x = i as f64 / 1000.0;
            let kc = kappa_c_at(x);
            assert!(
                kc.is_finite() && kc >= 0.0,
                "κ_c({}) = {} should be finite and ≥ 0",
                x,
                kc
            );
        }
    }

    /// Paper example is below κ_c under best-case L, Transitional under ternary.
    #[test]
    fn test_regime_paper() {
        let a: Vec<u64> = (1..=16).map(|i| i * 16).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let dr = DensityRegime::new(a.iter().copied(), e_mid);
        assert!(
            dr.best_case_kappa < dr.best_case_kappa_c,
            "best-case κ < κ_c: κ={}, κ_c={}",
            dr.best_case_kappa,
            dr.best_case_kappa_c
        );
        assert!(!dr.is_sparse(), "paper example is not sparse");
    }

    #[test]
    fn test_regime_large_n() {
        let a: Vec<u64> = (1..=100).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let dr = DensityRegime::new(a.iter().copied(), e_mid);
        assert!(!dr.is_sparse(), "N=100 should not be sparse");
        assert!(
            dr.best_case_kappa < 0.1,
            "best-case κ should be small for large N: {}",
            dr.best_case_kappa
        );
    }
}
