//! Unified return type for the four ambiguity-counting paths: brute force, sparse
//! convolution, Sasamoto saddle-point, and radix mappings.

use crate::count::oracle::BruteError;
use crate::count::sumset::Bound;

/// Count of ambiguity-producing interpretations of a transaction.
///
/// Generalizes two paper-distinct objects:
/// - `W(E) = #{S ⊆ A : ΣS = E}` (Sasamoto cond-mat/0106125) for `w_brute`/`w_sparse`/`w_sasamoto`
/// - `Σ k × m!` equivalent mappings (Notebook/Maurer) for `radix_mappings`
///
/// Both quantify alternative interpretations indistinguishable to an adversary.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Ambiguity {
    /// Exact count: full enumeration or untruncated sparse conv.
    Exact(u128),
    /// True count is ≥ this; sparse conv truncated.
    LowerBound(u128),
    /// `ln(W)` via saddle-point. Approximation, never a strict bound. The stored value
    /// must be finite; use [`Self::log_approx`] or `Self::from(Some(_))` to construct
    /// safely (both filter NaN and ±∞ into [`Self::Unknown`]).
    LogApprox(f64),
    /// No estimate (N too large, regime gate, infeasible, etc).
    Unknown,
}

impl Ambiguity {
    /// `Self::LogApprox` only when `log_w` is finite; NaN/±Inf collapse to `Unknown`.
    #[must_use]
    pub fn log_approx(log_w: f64) -> Self {
        if log_w.is_finite() {
            Self::LogApprox(log_w)
        } else {
            Self::Unknown
        }
    }

    /// Strict lower bound on the true count; `None` for approximations.
    #[must_use]
    pub fn lower_bound_count(&self) -> Option<u128> {
        match self {
            Self::Exact(n) | Self::LowerBound(n) => Some(*n),
            Self::LogApprox(_) | Self::Unknown => None,
        }
    }

    /// `ln(n)` for non-zero count variants, the stored value for `LogApprox`, and `None`
    /// for `Unknown`, `Exact(0)`, or `LowerBound(0)` (count 0 means unreachable, log undefined).
    #[must_use]
    pub fn log(&self) -> Option<f64> {
        match self {
            Self::Exact(n) | Self::LowerBound(n) if *n > 0 => Some((*n as f64).ln()),
            Self::LogApprox(lw) => Some(*lw),
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }

    #[must_use]
    pub const fn is_lower_bound(&self) -> bool {
        matches!(self, Self::LowerBound(_))
    }

    #[must_use]
    pub const fn is_approx(&self) -> bool {
        matches!(self, Self::LogApprox(_))
    }

    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

impl From<(u128, Bound)> for Ambiguity {
    fn from((count, bound): (u128, Bound)) -> Self {
        match bound {
            Bound::Exact => Self::Exact(count),
            Bound::LowerBound => Self::LowerBound(count),
        }
    }
}

impl From<Result<u128, BruteError>> for Ambiguity {
    fn from(r: Result<u128, BruteError>) -> Self {
        match r {
            Ok(n) => Self::Exact(n),
            Err(_) => Self::Unknown,
        }
    }
}

/// Maps `None` and non-finite inputs (NaN, ±∞) to [`Self::Unknown`]. `NEG_INFINITY` from
/// an approximation cannot upgrade to "unreachable" because approximations are never
/// trusted as strict bounds; callers that have exact 0 counts should construct
/// [`Self::Exact(0)`] directly instead.
impl From<Option<f64>> for Ambiguity {
    fn from(o: Option<f64>) -> Self {
        match o {
            Some(x) if x.is_finite() => Self::LogApprox(x),
            _ => Self::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_approx_filters_non_finite() {
        assert_eq!(Ambiguity::log_approx(1.5), Ambiguity::LogApprox(1.5));
        assert_eq!(Ambiguity::log_approx(f64::NAN), Ambiguity::Unknown);
        assert_eq!(Ambiguity::log_approx(f64::INFINITY), Ambiguity::Unknown);
        assert_eq!(Ambiguity::log_approx(f64::NEG_INFINITY), Ambiguity::Unknown);
    }

    #[test]
    fn lower_bound_count_only_for_count_variants() {
        assert_eq!(Ambiguity::Exact(42).lower_bound_count(), Some(42));
        assert_eq!(Ambiguity::LowerBound(42).lower_bound_count(), Some(42));
        assert_eq!(Ambiguity::LogApprox(3.5).lower_bound_count(), None);
        assert_eq!(Ambiguity::Unknown.lower_bound_count(), None);
    }

    #[test]
    fn log_unifies_count_and_approx() {
        let exact = Ambiguity::Exact(8);
        assert!((exact.log().unwrap() - (8f64).ln()).abs() < 1e-12);
        assert_eq!(Ambiguity::Exact(0).log(), None);
        assert_eq!(Ambiguity::LogApprox(2.5).log(), Some(2.5));
        assert_eq!(Ambiguity::Unknown.log(), None);
    }

    #[test]
    fn from_count_bound_tuple() {
        assert_eq!(
            Ambiguity::from((10u128, Bound::Exact)),
            Ambiguity::Exact(10)
        );
        assert_eq!(
            Ambiguity::from((10u128, Bound::LowerBound)),
            Ambiguity::LowerBound(10)
        );
    }

    #[test]
    fn from_brute_result() {
        assert_eq!(Ambiguity::from(Ok::<_, BruteError>(7)), Ambiguity::Exact(7));
        assert_eq!(
            Ambiguity::from(Err::<u128, _>(BruteError::TooLarge)),
            Ambiguity::Unknown
        );
        assert_eq!(
            Ambiguity::from(Err::<u128, _>(BruteError::SumOverflow)),
            Ambiguity::Unknown
        );
    }

    #[test]
    fn from_option_f64_filters_non_finite() {
        assert_eq!(Ambiguity::from(Some(2.5)), Ambiguity::LogApprox(2.5));
        assert_eq!(Ambiguity::from(None::<f64>), Ambiguity::Unknown);
        assert_eq!(Ambiguity::from(Some(f64::NAN)), Ambiguity::Unknown);
        assert_eq!(Ambiguity::from(Some(f64::NEG_INFINITY)), Ambiguity::Unknown);
    }

    use proptest::prelude::*;

    proptest! {
        /// `From<(u128, Bound)>` always lands on a count variant matching the bound.
        #[test]
        fn from_count_bound_round_trips(n: u128, lower: bool) {
            let bound = if lower { Bound::LowerBound } else { Bound::Exact };
            let w = Ambiguity::from((n, bound));
            prop_assert_eq!(w.lower_bound_count(), Some(n));
            prop_assert_eq!(w.is_exact(), !lower);
            prop_assert!(!w.is_unknown());
        }

        /// `Ambiguity::Exact(n).log() == ln(n)` for n > 0; same for `LowerBound`.
        #[test]
        fn log_matches_ln_for_count_variants(n in 1u128..=1_000_000_000) {
            let exact = Ambiguity::Exact(n);
            let lower = Ambiguity::LowerBound(n);
            let expected = (n as f64).ln();
            let de = exact.log().unwrap();
            let dl = lower.log().unwrap();
            prop_assert!((de - expected).abs() < 1e-9);
            prop_assert!((dl - expected).abs() < 1e-9);
        }

        /// `log_approx` is the identity on finite f64 and `Unknown` on the rest.
        #[test]
        fn log_approx_construction_invariant(x in proptest::num::f64::ANY) {
            let w = Ambiguity::log_approx(x);
            if x.is_finite() {
                prop_assert_eq!(w, Ambiguity::LogApprox(x));
                prop_assert_eq!(w.log(), Some(x));
            } else {
                prop_assert_eq!(w, Ambiguity::Unknown);
                prop_assert_eq!(w.log(), None);
            }
        }

        /// `lower_bound_count` is Some iff variant is Exact or LowerBound.
        #[test]
        fn lower_bound_count_only_on_counts(n: u128, lw: f64) {
            prop_assert_eq!(Ambiguity::Exact(n).lower_bound_count(), Some(n));
            prop_assert_eq!(Ambiguity::LowerBound(n).lower_bound_count(), Some(n));
            let approx_lb = Ambiguity::log_approx(lw).lower_bound_count();
            prop_assert_eq!(approx_lb, None);
            prop_assert_eq!(Ambiguity::Unknown.lower_bound_count(), None);
        }
    }
}
