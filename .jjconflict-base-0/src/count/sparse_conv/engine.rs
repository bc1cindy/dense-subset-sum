//! Las Vegas execution engine: budget-bounded loop over [`Phase`] schedules.

use super::field::Field;
use super::hash::LinearHash;
use super::result::{Convolution, finish};
use super::{
    SplitMix, bucketed_recover, bucketed_recover_residual, max_index, merge_max, sample_prime_in,
};
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub(super) enum Phase {
    Linear { m: usize, iters: u64 },
    PrimeResidual { m_prime: u64, iters: u64 },
    Checkpoint,
}

pub(super) struct LasVegasRun<'a, F: Field> {
    a: &'a [(u64, u64)],
    b: &'a [(u64, u64)],
    target: u128,
    n_max: u64,
    rng: SplitMix,
    acc: HashMap<u64, u64>,
    budget: u32,
    _field: PhantomData<F>,
}

impl<'a, F: Field> LasVegasRun<'a, F> {
    pub(super) fn new(
        a: &'a [(u64, u64)],
        b: &'a [(u64, u64)],
        target: u128,
        seed: u64,
        budget: u32,
    ) -> Self {
        Self {
            a,
            b,
            target,
            n_max: max_index(a)
                .unwrap_or(0)
                .saturating_add(max_index(b).unwrap_or(0)),
            rng: SplitMix::new(seed),
            acc: HashMap::new(),
            budget,
            _field: PhantomData,
        }
    }

    /// Linear-hash phase; max-fold into `acc`.
    fn run_iters(&mut self, m: usize, iters: u64) -> bool {
        for _ in 0..iters {
            if self.budget == 0 {
                return true;
            }
            self.budget -= 1;
            let h = LinearHash::new(self.rng.next(), m);
            merge_max(
                &mut self.acc,
                bucketed_recover::<F>(self.a, self.b, &h),
                self.n_max,
            );
        }
        false
    }

    /// Prime-hash residual phase; saturating-add into `acc`.
    fn run_residual_iters(&mut self, m_prime: u64, iters: u64) -> bool {
        for _ in 0..iters {
            if self.budget == 0 {
                return true;
            }
            self.budget -= 1;
            let p = sample_prime_in(m_prime, m_prime.saturating_mul(2), &mut self.rng);
            let c_iter = self.acc.iter().map(|(&k, &v)| (k, v));
            let updates = bucketed_recover_residual::<F, _>(self.a, self.b, c_iter, p);
            for (idx, c) in updates {
                if idx > self.n_max {
                    continue;
                }
                let entry = self.acc.entry(idx).or_insert(0);
                *entry = entry.saturating_add(c);
            }
        }
        false
    }

    fn is_complete(&self) -> bool {
        let total: u128 = self.acc.values().map(|&v| u128::from(v)).sum();
        total == self.target
    }

    fn into_result(self, terminated: bool) -> Convolution {
        finish(self.acc, self.target, terminated)
    }

    pub(super) fn execute(mut self, schedule: impl IntoIterator<Item = Phase>) -> Convolution {
        for phase in schedule {
            match phase {
                Phase::Linear { m, iters } => {
                    if self.run_iters(m, iters) {
                        return self.into_result(false);
                    }
                }
                Phase::PrimeResidual { m_prime, iters } => {
                    if self.run_residual_iters(m_prime, iters) {
                        return self.into_result(false);
                    }
                }
                Phase::Checkpoint => {
                    if self.is_complete() {
                        return self.into_result(true);
                    }
                }
            }
        }
        self.into_result(false)
    }
}
