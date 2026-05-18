//! Iteration schedules for BFN Algorithms 1, 4, 6.

use super::MAX_LOG_M;
use super::engine::Phase;

/// Algorithm 1 (Theorem 2): outer m = 2, 4, ..., `2^MAX_LOG_M` with `2 log m` linear-hash iters.
#[cfg(test)]
pub(super) fn alg1_schedule() -> impl Iterator<Item = Phase> {
    (1..=MAX_LOG_M).flat_map(|log_m| {
        let m = 1usize << log_m;
        let iters = u64::from(2 * log_m);
        [Phase::Linear { m, iters }, Phase::Checkpoint]
    })
}

/// Algorithm 4 (Lemma 19): tail-bounded with `µ·2^(ν/(1+ε))` repeats.
#[cfg(test)]
pub(super) fn alg4_schedule(epsilon: f64) -> impl Iterator<Item = Phase> {
    let denom = 1.0 + epsilon;
    (0..=MAX_LOG_M).flat_map(move |mu| {
        (0..=mu)
            .map(move |nu| {
                let m = 1usize << (mu - nu);
                // µ·2^(ν/(1+ε)) collapses to 0 at µ=0.
                let raw = if mu == 0 {
                    0.0
                } else {
                    f64::from(mu) * 2f64.powf(f64::from(nu) / denom)
                };
                Phase::Linear {
                    m,
                    // raw < MAX_LOG_M · 2^MAX_LOG_M ≈ 2e8, fits u64.
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    iters: raw.ceil() as u64,
                }
            })
            .chain(std::iter::once(Phase::Checkpoint))
    })
}

/// Algorithm 6 (Lemma 22): `3 log log n` linear-hash iters then `2 log m` prime-residual iters.
pub(super) fn alg6_schedule(log_n: usize, phase1_iters: u64) -> impl Iterator<Item = Phase> {
    (1..=MAX_LOG_M).flat_map(move |log_m| {
        let m = 1usize << log_m;
        let m_prime = (m / log_n).max(2) as u64;
        let phase2_iters = 2 * u64::from(log_m);
        [
            Phase::Linear {
                m,
                iters: phase1_iters,
            },
            Phase::PrimeResidual {
                m_prime,
                iters: phase2_iters,
            },
            Phase::Checkpoint,
        ]
    })
}
