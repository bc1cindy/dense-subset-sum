//! Las Vegas algorithm selection + sparse-conv dispatch (Bringmann/Fischer/Nakos arXiv:2107.07625).

use crate::count::sparse_conv::{Convolution, Field, Goldilocks, convolve_seeded_hybrid};
#[cfg(test)]
use crate::count::sparse_conv::{convolve_seeded, convolve_seeded_eps};
use std::marker::PhantomData;
use std::num::NonZeroUsize;

/// Construction policy; mutate via `with_*` builders.
#[derive(Clone, Copy, Debug)]
pub struct GradedSumsetBudget<F: Field = Goldilocks> {
    max_size: NonZeroUsize,
    las_vegas: LasVegas,
    max_inner_iters: u32,
    _field: PhantomData<F>,
}

impl<F: Field> GradedSumsetBudget<F> {
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_size: NonZeroUsize::MAX,
            las_vegas: LasVegas::Lemma22,
            max_inner_iters: u32::MAX,
            _field: PhantomData,
        }
    }

    #[must_use]
    pub const fn with_max_size(self, max_size: NonZeroUsize) -> Self {
        Self { max_size, ..self }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn with_las_vegas(self, las_vegas: LasVegas) -> Self {
        Self { las_vegas, ..self }
    }

    /// `u32::MAX` is unbounded.
    #[must_use]
    pub const fn with_max_inner_iters(self, max_inner_iters: u32) -> Self {
        Self {
            max_inner_iters,
            ..self
        }
    }

    #[must_use]
    pub const fn max_size(&self) -> usize {
        self.max_size.get()
    }

    #[must_use]
    pub(crate) const fn las_vegas(&self) -> LasVegas {
        self.las_vegas
    }

    #[must_use]
    pub const fn max_inner_iters(&self) -> u32 {
        self.max_inner_iters
    }
}

impl<F: Field> Default for GradedSumsetBudget<F> {
    fn default() -> Self {
        Self {
            max_size: DEFAULT_MAX_SIZE,
            las_vegas: LasVegas::Lemma22,
            max_inner_iters: u32::MAX,
            _field: PhantomData,
        }
    }
}

/// `Lemma22` is production; `Theorem2`/`Lemma19` test-only cross-validation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub(crate) enum LasVegas {
    #[cfg(test)]
    Theorem2,
    #[cfg(test)]
    Lemma19 { epsilon: f64 },
    /// O(t·log t·log log t) hybrid; fastest at CoinJoin scale.
    Lemma22,
}

impl std::fmt::Display for LasVegas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lemma22 => write!(f, "Lemma 22 (Algorithm 6)"),
            #[cfg(test)]
            Self::Theorem2 => write!(f, "Theorem 2 (Algorithm 1)"),
            #[cfg(test)]
            Self::Lemma19 { epsilon } => write!(f, "Lemma 19 (Algorithm 4, ε = {epsilon})"),
        }
    }
}

/// Distributed per-bucket (`max_size / (max_degree + 1)`) at cap time.
pub(crate) const DEFAULT_MAX_SIZE: NonZeroUsize = NonZeroUsize::new(1 << 16).unwrap();

pub(crate) const DEFAULT_CONV_SEED: u64 = 0xBC1B_57C0_FFEE_BABE;

pub(crate) fn dispatch_sparse<F: Field>(
    av: &[(u64, u64)],
    bv: &[(u64, u64)],
    seed: u64,
    max_inner_iters: u32,
    las_vegas: LasVegas,
) -> Convolution {
    match las_vegas {
        LasVegas::Lemma22 => convolve_seeded_hybrid::<F>(av, bv, seed, max_inner_iters),
        #[cfg(test)]
        LasVegas::Theorem2 => convolve_seeded::<F>(av, bv, seed, max_inner_iters),
        #[cfg(test)]
        LasVegas::Lemma19 { epsilon } => {
            convolve_seeded_eps::<F>(av, bv, seed, epsilon, max_inner_iters)
        }
    }
}
