/// `LowerBound` is absorbing under [`Bound::join`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bound {
    Exact,
    LowerBound,
}

impl std::fmt::Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact => f.write_str("exact"),
            Self::LowerBound => f.write_str("lower-bound"),
        }
    }
}

impl Bound {
    #[must_use]
    pub(crate) fn join(self, other: Self) -> Self {
        match (self, other) {
            (Bound::Exact, Bound::Exact) => Bound::Exact,
            _ => Bound::LowerBound,
        }
    }
}

/// Distinguishes pinned/exact counts from possibly-truncated zeros.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Count {
    Confirmed(u32),
    /// Lower bound (sumset truncated).
    Truncated(u32),
    /// Reliably 0.
    Absent,
    /// 0 from a truncated sumset; not reliable.
    Unknown,
}

pub(crate) const fn classify(visible: u32, truncated: bool) -> Count {
    match (visible, truncated) {
        (0, false) => Count::Absent,
        (0, true) => Count::Unknown,
        (n, false) => Count::Confirmed(n),
        (n, true) => Count::Truncated(n),
    }
}

impl Count {
    /// Lower bound on true count; 0 for `Absent`/`Unknown`.
    #[must_use]
    pub const fn visible(&self) -> u32 {
        match self {
            Count::Confirmed(n) | Count::Truncated(n) => *n,
            Count::Absent | Count::Unknown => 0,
        }
    }

    #[must_use]
    pub const fn is_exact(&self) -> bool {
        matches!(self, Count::Confirmed(_) | Count::Absent)
    }

    #[must_use]
    pub const fn bound(&self) -> Bound {
        match self {
            Count::Confirmed(_) | Count::Absent => Bound::Exact,
            Count::Truncated(_) | Count::Unknown => Bound::LowerBound,
        }
    }
}

impl std::fmt::Display for Count {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Count::Confirmed(n) => write!(f, "{n} (exact)"),
            Count::Truncated(n) => write!(f, "≥ {n} (lower-bound)"),
            Count::Absent => f.write_str("0 (exact)"),
            Count::Unknown => f.write_str("? (lower-bound, target not pinned)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_join_is_lowerbound_absorbing() {
        assert_eq!(Bound::Exact.join(Bound::Exact), Bound::Exact);
        assert_eq!(Bound::Exact.join(Bound::LowerBound), Bound::LowerBound);
        assert_eq!(Bound::LowerBound.join(Bound::Exact), Bound::LowerBound);
        assert_eq!(Bound::LowerBound.join(Bound::LowerBound), Bound::LowerBound);
    }

    #[test]
    fn bound_display() {
        assert_eq!(Bound::Exact.to_string(), "exact");
        assert_eq!(Bound::LowerBound.to_string(), "lower-bound");
    }

    #[test]
    fn classify_maps_4_quadrants() {
        assert_eq!(classify(0, false), Count::Absent);
        assert_eq!(classify(0, true), Count::Unknown);
        assert_eq!(classify(7, false), Count::Confirmed(7));
        assert_eq!(classify(7, true), Count::Truncated(7));
    }

    #[test]
    fn count_visible_returns_zero_for_no_count_variants() {
        assert_eq!(Count::Confirmed(5).visible(), 5);
        assert_eq!(Count::Truncated(5).visible(), 5);
        assert_eq!(Count::Absent.visible(), 0);
        assert_eq!(Count::Unknown.visible(), 0);
    }

    #[test]
    fn count_is_exact_only_confirmed_or_absent() {
        assert!(Count::Confirmed(5).is_exact());
        assert!(Count::Absent.is_exact());
        assert!(!Count::Truncated(5).is_exact());
        assert!(!Count::Unknown.is_exact());
    }

    #[test]
    fn count_bound_matches_variant() {
        assert_eq!(Count::Confirmed(5).bound(), Bound::Exact);
        assert_eq!(Count::Absent.bound(), Bound::Exact);
        assert_eq!(Count::Truncated(5).bound(), Bound::LowerBound);
        assert_eq!(Count::Unknown.bound(), Bound::LowerBound);
    }

    #[test]
    fn count_display() {
        assert_eq!(Count::Confirmed(42).to_string(), "42 (exact)");
        assert_eq!(Count::Truncated(7).to_string(), "≥ 7 (lower-bound)");
        assert_eq!(Count::Absent.to_string(), "0 (exact)");
        assert_eq!(
            Count::Unknown.to_string(),
            "? (lower-bound, target not pinned)"
        );
    }
}
