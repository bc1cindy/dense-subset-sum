//! Sasamoto density regime: dense (κ < `κ_c`) ⇒ saddle-point reliable; sparse otherwise.
//! κ = log₂(L)/N; `κ_c` from eq (4.3).

use std::f64::consts::LN_2;

pub const MAX_MONEY: u64 = 2_100_000_000_000_000;

/// Number of steps for [`trapezoidal_integral`] when computing `κ_c`.
/// 64 keeps Sasamoto eq (4.3) integrals within 1e-6 of the analytic limit.
const TRAPEZOIDAL_STEPS: usize = 64;

/// Bracket for [`find_alpha`]. Wider than the saddle's reach for
/// `x ∈ [ASYMPTOTIC_THRESHOLD, 0.5)`; outside that, the asymptotic short
/// circuit avoids precision loss.
const ALPHA_BRACKET: (f64, f64) = (-1000.0, 1000.0);

/// Below this `x = E/(N·L)`, the asymptotic `κ_c ≈ 2√x / ln 2` is more
/// accurate than [`find_alpha`]'s bisect, whose target value vanishes
/// faster than f64 resolution.
const ASYMPTOTIC_THRESHOLD: f64 = 1e-3;

/// κ = log₂(L)/N. `None` when `n == 0` or `l == 0`.
#[must_use]
pub fn kappa(l: u64, n: usize) -> Option<f64> {
    if n == 0 || l == 0 {
        return None;
    }
    Some((l as f64).log2() / n as f64)
}

/// κ at L = `MAX_MONEY/N` (A-independent worst case).
#[must_use]
pub fn worst_case_kappa(n: usize) -> Option<f64> {
    kappa(MAX_MONEY / (n as u64).max(1), n)
}

/// L candidates for κ = log₂(L)/N. [`Bracket::new`] uses only [`L::Max`] (best
/// case) and [`L::MaxMoneyOverN`] (worst case); the other variants are exposed
/// for downstream regime analysis or alternative bracket choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum L {
    Max,
    MaxMoney,
    MaxMoneyOverN,
    Sum,
    MaxSquared,
    SumOfSquares,
    SumSquared,
}

impl std::fmt::Display for L {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Max => "max(A)",
            Self::MaxMoney => "MAX_MONEY",
            Self::MaxMoneyOverN => "MAX_MONEY/N",
            Self::Sum => "Σ A",
            Self::MaxSquared => "max(A)²",
            Self::SumOfSquares => "Σ aᵢ²",
            Self::SumSquared => "(Σ A)²",
        };
        f.write_str(s)
    }
}

impl L {
    #[must_use]
    pub const fn all() -> &'static [L] {
        &[
            Self::Max,
            Self::MaxMoney,
            Self::MaxMoneyOverN,
            Self::Sum,
            Self::MaxSquared,
            Self::SumOfSquares,
            Self::SumSquared,
        ]
    }

    /// Saturating at `u64::MAX` on overflow; 0 on empty.
    pub fn value(self, a: &[u64]) -> u64 {
        match self {
            Self::Max => a.iter().copied().max().unwrap_or(0),
            Self::MaxMoney => MAX_MONEY,
            Self::MaxMoneyOverN => {
                if a.is_empty() {
                    0
                } else {
                    MAX_MONEY / a.len() as u64
                }
            }
            Self::Sum => a.iter().copied().fold(0u64, u64::saturating_add),
            Self::MaxSquared => {
                let m = a.iter().copied().max().unwrap_or(0);
                m.saturating_mul(m)
            }
            Self::SumOfSquares => a
                .iter()
                .copied()
                .map(|x| x.saturating_mul(x))
                .fold(0u64, u64::saturating_add),
            Self::SumSquared => {
                let s = a.iter().copied().fold(0u64, u64::saturating_add);
                s.saturating_mul(s)
            }
        }
    }
}

/// Regime classification under a specific L choice. Single-L variant of
/// [`Bracket::new`], which uses [`L::Max`] (best) and [`L::MaxMoneyOverN`]
/// (worst) jointly. Useful for exploring how different L choices shift the
/// dense/sparse boundary for the same `(set, e_target)`.
///
/// Returns:
/// - `Some(Regime::Dense)` when `κ < κ_c` at the chosen L
/// - `Some(Regime::Sparse)` when `κ ≥ κ_c`
/// - `None` when L, κ, or κ_c can't be computed (empty set, L=0, x∉(0,1])
///
/// Never returns `Transitional` since that requires comparing two L choices.
#[must_use]
pub fn regime_at_l(set: &[u64], e_target: u64, l_choice: L) -> Option<Regime> {
    let l_val = l_choice.value(set);
    let n = set.len();
    let k = kappa(l_val, n)?;
    let kc = kappa_c(e_target, n, l_val)?;
    Some(if k < kc {
        Regime::Dense
    } else {
        Regime::Sparse
    })
}

/// `κ_c` per Sasamoto eq (4.2, 4.3). `None` when x = E/(N·L) ∉ (0, 1].
#[must_use]
pub fn kappa_c(e: u64, n: usize, l: u64) -> Option<f64> {
    if n == 0 || l == 0 {
        return None;
    }
    let x = e as f64 / (n as f64 * l as f64);
    if !(0.0 < x && x <= 1.0) {
        return None;
    }
    Some(kappa_c_at(x))
}

/// Dense iff κ < `κ_c` at worst-case L; Sparse iff κ ≥ `κ_c` at best-case L; else Transitional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Regime {
    Dense,
    Transitional,
    Sparse,
}

impl std::fmt::Display for Regime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Dense => "Dense",
            Self::Transitional => "Transitional",
            Self::Sparse => "Sparse",
        };
        f.write_str(s)
    }
}

/// `(best, worst)` pair. best = L = max(A); worst = L = `MAX_MONEY/N`.
///
/// Exposed pub because returned by [`Bracket::kappa`] and [`Bracket::kappa_c`].
#[derive(Clone, Copy, Debug)]
pub struct Interval<T> {
    best: T,
    worst: T,
}

impl<T: Copy> Interval<T> {
    #[must_use]
    pub(crate) const fn new(best: T, worst: T) -> Self {
        Self { best, worst }
    }

    #[must_use]
    pub const fn best(&self) -> T {
        self.best
    }

    #[must_use]
    pub const fn worst(&self) -> T {
        self.worst
    }
}

#[must_use]
pub struct Bracket {
    kappa: Interval<f64>,
    kappa_c: Interval<f64>,
    regime: Regime,
}

impl Bracket {
    /// Build the `κ/κ_c` bracket pair plus regime classification for `(A, E)`.
    ///
    /// # Errors
    ///
    /// Returns `None` when `a` is empty or when `e_target` is outside the
    /// feasibility window `(0, N · max(A)]` (no subset can sum there).
    pub fn new<I>(a: I, e_target: u64) -> Option<Self>
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator,
    {
        let a = a.into_iter();
        let n = a.len();
        let l_best = a.max()?;
        let l_worst = MAX_MONEY / (n as u64).max(1);

        let kappa = Interval::new(kappa(l_best, n)?, kappa(l_worst, n)?);
        let kappa_c = Interval::new(
            kappa_c(e_target, n, l_best)?,
            kappa_c(e_target, n, l_worst)?,
        );
        let regime = if kappa.worst < kappa_c.worst {
            Regime::Dense
        } else if kappa.best >= kappa_c.best {
            Regime::Sparse
        } else {
            Regime::Transitional
        };
        Some(Self {
            kappa,
            kappa_c,
            regime,
        })
    }

    #[must_use]
    pub fn regime(&self) -> Regime {
        self.regime
    }

    #[must_use]
    pub fn kappa(&self) -> Interval<f64> {
        self.kappa
    }

    #[must_use]
    pub fn kappa_c(&self) -> Interval<f64> {
        self.kappa_c
    }
}

impl std::fmt::Display for Bracket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "κ ∈ [{:.4}, {:.4}], κ_c ∈ [{:.4}, {:.4}], regime = {}",
            self.kappa.best,
            self.kappa.worst,
            self.kappa_c.best,
            self.kappa_c.worst,
            self.regime(),
        )
    }
}

/// `κ_c(x)` per eq (4.2, 4.3); symmetric about 1/2, peaks `κ_c(1/4)=1`.
fn kappa_c_at(x: f64) -> f64 {
    let x = if x > 0.5 { 1.0 - x } else { x };
    // eq (4.3) at x = 1/2: α → -∞, integrand cancels.
    if x == 0.5 {
        return 0.0;
    }
    if x < ASYMPTOTIC_THRESHOLD {
        return 2.0 * x.sqrt() / LN_2;
    }
    let alpha = find_alpha(x);
    let integral = trapezoidal_integral(
        |s| (1.0 + (-alpha * s).exp()).ln(),
        0.0,
        1.0,
        TRAPEZOIDAL_STEPS,
    );
    (integral + alpha * x) / LN_2
}

/// Inverts ∫₀¹ s/(1+e^(α·s)) ds = x; integrand monotone in α.
fn find_alpha(x: f64) -> f64 {
    let (lo, hi) = ALPHA_BRACKET;
    let f = |alpha: f64| {
        trapezoidal_integral(
            |s| s / (1.0 + (alpha * s).exp()),
            0.0,
            1.0,
            TRAPEZOIDAL_STEPS,
        )
    };
    assert!(
        f(lo) >= x && f(hi) <= x,
        "find_alpha: x = {} outside bracket [f({})={}, f({})={}]",
        x,
        lo,
        f(lo),
        hi,
        f(hi),
    );
    crate::count::numeric::bisect(f, lo, hi, x, 200, 1e-12)
}

fn trapezoidal_integral<F: Fn(f64) -> f64>(integrand: F, lo: f64, hi: f64, steps: usize) -> f64 {
    let step = (hi - lo) / steps as f64;
    let interior: f64 = (1..steps).map(|i| integrand(lo + i as f64 * step)).sum();
    (f64::midpoint(integrand(lo), integrand(hi)) + interior) * step
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn regime_at_l_returns_none_on_empty() {
        assert!(regime_at_l(&[], 10, L::Max).is_none());
    }

    #[test]
    fn regime_at_l_returns_none_on_out_of_range_e() {
        let a = [10u64, 20, 30];
        // x = E/(N·L); E=0 → x=0 ∉ (0,1] → None
        assert!(regime_at_l(&a, 0, L::Max).is_none());
    }

    #[test]
    fn regime_at_l_dense_for_typical_dense_input() {
        // (1..=20)*100 with E=sum/2 is classically dense at L=Max.
        let a: Vec<u64> = (1..=20).map(|i| i * 100).collect();
        let e = a.iter().sum::<u64>() / 2;
        assert_eq!(regime_at_l(&a, e, L::Max), Some(Regime::Dense));
    }

    #[test]
    fn regime_at_l_sparse_at_worst_case_l() {
        // L::MaxMoneyOverN inflates L massively → κ huge → sparse classification.
        let a: Vec<u64> = (1..=8).collect();
        let e = a.iter().sum::<u64>() / 2;
        assert_eq!(regime_at_l(&a, e, L::MaxMoneyOverN), Some(Regime::Sparse));
    }

    #[test]
    fn regime_at_l_never_returns_transitional() {
        // Single-L classification cannot bracket; only Dense or Sparse.
        let a: Vec<u64> = (1..=15).map(|i| i * 100).collect();
        let e = a.iter().sum::<u64>() / 3;
        for &l in L::all() {
            if let Some(r) = regime_at_l(&a, e, l) {
                assert!(!matches!(r, Regime::Transitional));
            }
        }
    }

    /// Paper's A (N=16); `best_case` uses max(A)=239 vs paper's L=256.
    #[test]
    fn best_case_kappa_paper_example() {
        let a: Vec<u64> = vec![
            218, 13, 227, 193, 70, 134, 89, 198, 205, 147, 227, 190, 27, 239, 192, 131,
        ];
        let e = a.iter().sum::<u64>() / 2;
        let k = Bracket::new(a.iter().copied(), e).unwrap().kappa.best;
        let expected = (239.0_f64).log2() / 16.0;
        assert!((k - expected).abs() < 1e-10, "expected {expected}, got {k}");
    }

    #[test]
    fn best_case_kappa_decreases_with_n() {
        let a10: Vec<u64> = (1..=10).map(|i| i * 100).collect();
        let a20: Vec<u64> = (1..=20).map(|i| i * 100).collect();
        let e10 = a10.iter().sum::<u64>() / 2;
        let e20 = a20.iter().sum::<u64>() / 2;
        let k10 = Bracket::new(a10.iter().copied(), e10).unwrap().kappa.best;
        let k20 = Bracket::new(a20.iter().copied(), e20).unwrap().kappa.best;
        assert!(k20 < k10, "κ should decrease: k10={k10}, k20={k20}");
    }

    #[test]
    fn density_regime_new_empty_returns_none() {
        assert!(Bracket::new(std::iter::empty::<u64>(), 1).is_none());
    }

    #[test]
    fn density_regime_new_e_outside_domain_returns_none() {
        let a = [10u64, 20, 30];
        assert!(Bracket::new(a.iter().copied(), 0).is_none());
        assert!(Bracket::new(a.iter().copied(), u64::MAX).is_none());
    }

    #[test]
    fn kappa_returns_none_for_invalid_inputs() {
        assert!(kappa(1024, 0).is_none());
        assert!(kappa(0, 10).is_none());
    }

    #[test]
    fn l_all_lists_every_variant() {
        let all = L::all();
        assert_eq!(all.len(), 7);
        for (i, &x) in all.iter().enumerate() {
            for &y in &all[i + 1..] {
                assert_ne!(x, y, "duplicate L variant in all()");
            }
        }
    }

    #[test]
    fn l_candidates_basic() {
        let a: Vec<u64> = vec![3, 5, 7];
        assert_eq!(L::Max.value(&a), 7);
        assert_eq!(L::MaxMoney.value(&a), MAX_MONEY);
        assert_eq!(L::MaxMoneyOverN.value(&a), MAX_MONEY / 3);
        assert_eq!(L::Sum.value(&a), 15);
        assert_eq!(L::MaxSquared.value(&a), 49);
        assert_eq!(L::SumOfSquares.value(&a), 9 + 25 + 49);
        assert_eq!(L::SumSquared.value(&a), 225);
    }

    #[test]
    fn l_candidates_empty() {
        let a: Vec<u64> = vec![];
        assert_eq!(L::Max.value(&a), 0);
        assert_eq!(L::MaxMoneyOverN.value(&a), 0);
        assert_eq!(L::Sum.value(&a), 0);
        assert_eq!(L::SumSquared.value(&a), 0);
    }

    /// Pipeline `L → κ → κ_c → regime` finite for every candidate.
    #[test]
    fn l_candidates_pipeline_per_candidate() {
        let a: Vec<u64> = (1..=20).map(|i| i * 1_000).collect();
        let n = a.len();
        let e_target: u64 = a.iter().sum::<u64>() / 2;
        for &candidate in L::all() {
            let l = candidate.value(&a);
            assert!(l > 0, "{candidate:?}: L = 0 unexpected for non-empty a");
            let nl = (n as u128) * u128::from(l);
            assert!(
                nl >= u128::from(e_target),
                "{candidate:?}: N·L = {nl} < E = {e_target}, kappa_c would panic"
            );
            let k = kappa(l, n).unwrap();
            let kc = kappa_c(e_target, n, l).expect("nl >= e ensures domain valid");
            assert!(k.is_finite(), "{candidate:?}: κ = {k} not finite");
            assert!(
                kc.is_finite() && kc >= 0.0,
                "{candidate:?}: κ_c = {kc} invalid"
            );
            let _: Regime = if k < kc {
                Regime::Dense
            } else {
                Regime::Sparse
            };
        }
    }

    #[test]
    fn l_candidates_saturate_on_overflow() {
        let a: Vec<u64> = vec![u64::MAX];
        assert_eq!(L::MaxSquared.value(&a), u64::MAX);
        assert_eq!(L::SumOfSquares.value(&a), u64::MAX);
        assert_eq!(L::SumSquared.value(&a), u64::MAX);
    }

    #[test]
    fn kappa_c_midpoint() {
        let kc = kappa_c_at(0.25);
        assert!(
            kc > 0.0 && kc.is_finite(),
            "κ_c(0.25) = {kc} should be positive finite"
        );
    }

    #[test]
    fn kappa_c_symmetry() {
        let kc1 = kappa_c_at(0.2);
        let kc2 = kappa_c_at(0.8);
        assert!(
            (kc1 - kc2).abs() < 1e-4,
            "κ_c should be symmetric: κ_c(0.2)={kc1}, κ_c(0.8)={kc2}"
        );
    }

    /// Peaks at x=1/4 (α=0, `κ_c=1`), NOT at x=1/2 (paper Fig. 2 double-hump).
    #[test]
    fn kappa_c_peak_at_quartile() {
        let kc_quarter = kappa_c_at(0.25);
        let kc_edge = kappa_c_at(0.05);
        let kc_near_half = kappa_c_at(0.45);
        assert!(
            (kc_quarter - 1.0).abs() < 1e-3,
            "κ_c(1/4) should be ≈ 1 (α = 0), got {kc_quarter}",
        );
        assert!(
            kc_quarter > kc_edge,
            "κ_c(1/4) > κ_c(0.05): got {kc_quarter} vs {kc_edge}",
        );
        assert!(
            kc_quarter > kc_near_half,
            "κ_c(1/4) > κ_c(0.45) (valley toward 1/2): got {kc_quarter} vs {kc_near_half}",
        );
    }

    /// Paper Fig. 2 / eq (4.3): `κ_c(1/2)` = 0 (saddle α → -∞, integrand cancels).
    #[test]
    fn kappa_c_at_half_is_zero() {
        assert_eq!(kappa_c_at(0.5), 0.0);

        let a = [10u64; 8];
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let dr = Bracket::new(a.iter().copied(), e_mid).unwrap();
        assert_eq!(dr.kappa_c.best, 0.0);
        assert!(dr.kappa.best > 0.0);
        assert!(
            matches!(dr.regime(), Regime::Sparse),
            "equal-denom midpoint is sparse (κ > 0 = κ_c)"
        );
    }

    #[test]
    fn kappa_c_e_matches_x() {
        assert!((kappa_c(100, 10, 50).unwrap() - kappa_c_at(0.2)).abs() < 1e-12);
    }

    #[test]
    fn kappa_c_returns_none_outside_domain() {
        // x = 0 (E=0)
        assert!(kappa_c(0, 10, 50).is_none());
        // n = 0
        assert!(kappa_c(100, 0, 50).is_none());
        // l = 0
        assert!(kappa_c(100, 10, 0).is_none());
        // x > 1 (E too large)
        assert!(kappa_c(1000, 10, 50).is_none());
    }

    /// Whirlpool-style E = Σ A: x = 1, `κ_c(1)` = 0 per paper.
    #[test]
    fn kappa_c_equal_values_whirlpool() {
        let a: Vec<u64> = vec![100_000_000; 8];
        let e_total: u64 = a.iter().sum();
        let l = *a.iter().max().unwrap();

        assert_eq!(kappa_c(e_total, a.len(), l), Some(0.0));

        let dr = Bracket::new(a.iter().copied(), e_total).unwrap();
        assert_eq!(dr.kappa_c.best, 0.0);
        assert!(
            matches!(dr.regime(), Regime::Sparse),
            "equal-value tx at E=Σ: κ > 0 = κ_c"
        );
    }

    proptest! {
        /// f(find_alpha(x)) ≈ x within numerical domain.
        #[test]
        fn find_alpha_inverts_integral(x in 1e-4f64..(0.5 - 1e-4)) {
            let alpha = find_alpha(x);
            let f_at_alpha = trapezoidal_integral(
                |s| s / (1.0 + (alpha * s).exp()), 0.0, 1.0, TRAPEZOIDAL_STEPS);
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
            let kc = kappa_c(e, n, l).expect("e in (0, n·l] ensures domain");
            prop_assert!(kc.is_finite() && kc >= 0.0);
        }

        /// κ_c(x) = κ_c(1−x) (paper Fig. 2: symmetric about 1/2).
        #[test]
        fn kappa_c_symmetric_about_half(x in 1e-3f64..0.499) {
            let kc1 = kappa_c_at(x);
            let kc2 = kappa_c_at(1.0 - x);
            prop_assert!(
                (kc1 - kc2).abs() < 1e-4,
                "κ_c({}) = {}, κ_c({}) = {}",
                x, kc1, 1.0 - x, kc2
            );
        }

        /// max(A) ≤ Σ A ≤ (Σ A)²; κ monotone in log L.
        #[test]
        fn l_candidates_ordering(
            a in proptest::collection::vec(1u64..1_000_000, 1..50),
        ) {
            let n = a.len();
            let v_max = L::Max.value(&a);
            let v_sum = L::Sum.value(&a);
            let v_sum_sq = L::SumSquared.value(&a);
            prop_assert!(v_max <= v_sum, "max(A)={} > Σ A={}", v_max, v_sum);
            if v_sum_sq < u64::MAX {
                prop_assert!(v_sum <= v_sum_sq, "Σ A={} > (Σ A)²={}", v_sum, v_sum_sq);
            }
            if v_max >= 1 && v_sum >= v_max {
                let k_max = kappa(v_max, n).unwrap();
                let k_sum = kappa(v_sum, n).unwrap();
                prop_assert!(
                    k_max <= k_sum + 1e-12,
                    "κ(Max)={} > κ(Sum)={}", k_max, k_sum
                );
            }
        }

        /// Saturation arithmetic: every L candidate is total.
        #[test]
        fn l_candidates_total_on_any_input(
            a in proptest::collection::vec(0u64..u64::MAX, 0..30),
        ) {
            for &candidate in L::all() {
                let _ = candidate.value(&a);
            }
        }

        /// max(A) ≤ max(A)² when max(A) ≥ 1.
        #[test]
        fn l_max_le_max_squared(
            a in proptest::collection::vec(1u64..1_000_000, 1..50),
        ) {
            let v_max = L::Max.value(&a);
            let v_max_sq = L::MaxSquared.value(&a);
            if v_max_sq < u64::MAX {
                prop_assert!(v_max <= v_max_sq, "max(A)={} > max(A)²={}", v_max, v_max_sq);
            }
        }

        /// Σ A ≤ Σ aᵢ² when each aᵢ ≥ 1.
        #[test]
        fn l_sum_le_sum_of_squares(
            a in proptest::collection::vec(1u64..1_000_000, 1..50),
        ) {
            let v_sum = L::Sum.value(&a);
            let v_sumsq = L::SumOfSquares.value(&a);
            if v_sumsq < u64::MAX {
                prop_assert!(v_sum <= v_sumsq, "Σ A={} > Σ aᵢ²={}", v_sum, v_sumsq);
            }
        }

        /// Σ aᵢ² ≤ (Σ A)² for aᵢ ≥ 0.
        #[test]
        fn l_sum_of_squares_le_sum_squared(
            a in proptest::collection::vec(0u64..1_000_000, 1..50),
        ) {
            let v_sumsq = L::SumOfSquares.value(&a);
            let v_sum_sq = L::SumSquared.value(&a);
            if v_sum_sq < u64::MAX && v_sumsq < u64::MAX {
                prop_assert!(v_sumsq <= v_sum_sq, "Σ aᵢ²={} > (Σ A)²={}", v_sumsq, v_sum_sq);
            }
        }

        #[test]
        fn l_max_money_over_n_le_max_money(
            a in proptest::collection::vec(0u64..1_000_000, 1..100),
        ) {
            let v_per_n = L::MaxMoneyOverN.value(&a);
            let v_max_money = L::MaxMoney.value(&a);
            prop_assert!(v_per_n <= v_max_money, "MAX_MONEY/N={} > MAX_MONEY={}", v_per_n, v_max_money);
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
        let result = trapezoidal_integral(f64::exp, 0.0, 1.0, 256);
        assert!(
            (result - expected).abs() < 1e-5,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn trapezoidal_approximates_ln() {
        // ∫₁² ln(x) dx = 2·ln(2) - 1.
        let expected = 2.0 * 2.0_f64.ln() - 1.0;
        let result = trapezoidal_integral(f64::ln, 1.0, 2.0, 256);
        assert!(
            (result - expected).abs() < 1e-6,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn find_alpha_outside_domain_panics() {
        // x ≪ ASYMPTOTIC_THRESHOLD: f(hi) ≈ 0.5/hi² overshoots x, fails the
        // f(hi) ≤ x check. x ≈ 0.5: f(lo) approaches 0.5 from below but the
        // trapezoidal residual exceeds f64::EPSILON, fails the f(lo) ≥ x check.
        // Callers route through kappa_c_at, which short-circuits both regions.
        let unsupported = [1e-100, f64::EPSILON, f64::MIN_POSITIVE, 0.5 - f64::EPSILON];
        for x in unsupported {
            let result = std::panic::catch_unwind(|| find_alpha(x));
            assert!(result.is_err(), "find_alpha({x:e}) should panic");
        }
    }

    #[test]
    fn kappa_c_at_finite_across_range() {
        for i in 1..=999 {
            let x = f64::from(i) / 1000.0;
            let kc = kappa_c_at(x);
            assert!(
                kc.is_finite() && kc >= 0.0,
                "κ_c({x}) = {kc} should be finite and ≥ 0"
            );
        }
    }

    /// Paper example is below `κ_c` under best-case L, Transitional under ternary.
    #[test]
    fn regime_paper() {
        let a: Vec<u64> = (1..=16).map(|i| i * 16).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let dr = Bracket::new(a.iter().copied(), e_mid).unwrap();
        assert!(
            dr.kappa.best < dr.kappa_c.best,
            "best-case κ < κ_c: κ={}, κ_c={}",
            dr.kappa.best,
            dr.kappa_c.best
        );
        assert!(
            matches!(dr.regime(), Regime::Transitional),
            "paper example is Transitional"
        );
    }

    #[test]
    fn regime_large_n() {
        let a: Vec<u64> = (1..=100).collect();
        let e_mid: u64 = a.iter().sum::<u64>() / 2;
        let dr = Bracket::new(a.iter().copied(), e_mid).unwrap();
        assert!(
            matches!(dr.regime(), Regime::Transitional),
            "N=100 mid-target is Transitional"
        );
        assert!(
            dr.kappa.best < 0.1,
            "best-case κ should be small for large N: {}",
            dr.kappa.best
        );
    }

    /// N=100 equal coins at `E=MAX_MONEY/4`: `x_worst` lands at `κ_c` peak ≈ 1.0.
    #[test]
    fn regime_dense_high_n() {
        let n = 100usize;
        let e_target = MAX_MONEY / 4;
        let coin = e_target / n as u64;
        let a: Vec<u64> = vec![coin; n];
        let dr = Bracket::new(a.iter().copied(), e_target).unwrap();
        assert!(
            matches!(dr.regime(), Regime::Dense),
            "expected Dense, got κ={} κ_c={} (worst), κ={} κ_c={} (best)",
            dr.kappa.worst,
            dr.kappa_c.worst,
            dr.kappa.best,
            dr.kappa_c.best,
        );
    }
}
