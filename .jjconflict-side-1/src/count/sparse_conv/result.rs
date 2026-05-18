//! Convolution output type and termination tag.

use std::collections::HashMap;

/// Termination state of a sparse convolution run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Termination {
    /// Output is exact: every `(sum, count)` in support is the true value,
    /// and the support enumerates every pair with non-zero count.
    Complete,
    /// Output is a strict lower bound: support may be incomplete or counts
    /// may underestimate the truth (budget exhausted or precondition not met).
    LowerBound,
}

#[derive(Debug, Clone)]
#[must_use]
pub struct Convolution {
    pub(super) support: Vec<(u64, u64)>,
    pub(super) termination: Termination,
}

impl Convolution {
    pub(super) fn empty_complete() -> Self {
        Self {
            support: Vec::new(),
            termination: Termination::Complete,
        }
    }

    /// `‖A‖₁·‖B‖₁ ≥ P` would diverge mod-P from u128 target.
    pub(super) fn precondition_violated() -> Self {
        Self {
            support: Vec::new(),
            termination: Termination::LowerBound,
        }
    }

    #[must_use]
    pub fn support(&self) -> &[(u64, u64)] {
        &self.support
    }

    #[must_use]
    pub fn into_support(self) -> Vec<(u64, u64)> {
        self.support
    }

    #[must_use]
    pub fn termination(&self) -> Termination {
        self.termination
    }
}

pub(super) fn finish(acc: HashMap<u64, u64>, target: u128, terminated: bool) -> Convolution {
    let total: u128 = acc.values().map(|&v| u128::from(v)).sum();
    let termination = if terminated && total == target {
        Termination::Complete
    } else {
        Termination::LowerBound
    };
    Convolution {
        support: acc.into_iter().collect(),
        termination,
    }
}
