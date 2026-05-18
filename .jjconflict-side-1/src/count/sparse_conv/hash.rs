//! Multiply-shift linear hashing per Lemma 13 (Bringmann/Fischer/Nakos).

#[derive(Debug, Clone, Copy)]
pub struct LinearHash {
    a: u64,
    log_m: u32,
}

impl LinearHash {
    /// `a` forced odd: even multipliers degenerate to h ≡ 0.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `m` is a power of two and `m ≤ 2^32`.
    #[must_use]
    pub fn new(a: u64, m: usize) -> Self {
        debug_assert!(m.is_power_of_two(), "m must be a power of two");
        debug_assert!(m.trailing_zeros() <= 32, "m exceeds 2^32");
        Self {
            a: a | 1,
            log_m: m.trailing_zeros(),
        }
    }

    #[must_use]
    pub fn eval(&self, x: u64) -> usize {
        (u128::from(self.a.wrapping_mul(x)) >> (64 - self.log_m)) as usize
    }

    #[must_use]
    pub fn m(&self) -> usize {
        1usize << self.log_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn m_one_is_constant_zero() {
        let h = LinearHash::new(0xdead_beef, 1);
        for x in 0u64..1000 {
            assert_eq!(h.eval(x), 0);
        }
    }

    #[test]
    fn even_seed_becomes_odd_multiplier() {
        let h = LinearHash::new(0, 256);
        assert_ne!(h.eval(1u64 << 60), 0);
    }

    #[test]
    fn almost_additive_phi_subset_of_zero_and_minus_one() {
        let m = 256;
        let h = LinearHash::new(0x9E37_79B9_7F4A_7C15, m);
        for x in 0u64..200 {
            for y in 0u64..200 {
                let lhs = (h.eval(x) + h.eval(y)) % m;
                let rhs = h.eval(x.wrapping_add(y));
                let phi = (lhs + m - rhs) % m;
                assert!(phi == 0 || phi == m - 1, "unexpected phi: {phi}");
            }
        }
    }

    /// Lemma 13(2) uniform-difference: `Pr[h(x)−h(y) ≡ q (mod m)] ≤ 2/m`
    /// for any fixed `x ≠ y, q`. Empirical check over many random hashes.
    #[test]
    fn lemma_13_uniform_difference_rate() {
        let bucket_count = 256usize;
        let trials = 200_000;
        // Fix x, y, q; vary the hash multiplier.
        let probe_x: u64 = 0x1234_5678_9ABC_DEF0;
        let probe_y: u64 = 0x0FED_CBA9_8765_4321;
        let target_diff: usize = 73;
        let mut hits = 0;
        let mut rng_state: u64 = 0xC0FF_EEBA_DEAD_BEEF;
        for _ in 0..trials {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let hasher = LinearHash::new(rng_state, bucket_count);
            let diff = (hasher.eval(probe_x) + bucket_count - hasher.eval(probe_y)) % bucket_count;
            if diff == target_diff {
                hits += 1;
            }
        }
        let rate = f64::from(hits) / f64::from(trials);
        let bound = 2.0 / bucket_count as f64;
        assert!(
            rate < bound * 1.5,
            "rate {rate:.6} exceeds 1.5·(2/m) = {:.6}",
            bound * 1.5
        );
    }

    proptest! {
        #[test]
        fn prop_output_in_range(
            a: u64,
            log_m in 0u32..=10u32,
            x: u64,
        ) {
            let m = 1usize << log_m;
            let h = LinearHash::new(a, m);
            prop_assert!(h.eval(x) < m);
        }

        #[test]
        fn prop_deterministic(a: u64, log_m in 0u32..=10u32, x: u64) {
            let m = 1usize << log_m;
            let h = LinearHash::new(a, m);
            prop_assert_eq!(h.eval(x), h.eval(x));
        }

        #[test]
        fn prop_almost_additive(
            a: u64,
            log_m in 1u32..=10u32,
            x: u64,
            y: u64,
        ) {
            let m = 1usize << log_m;
            let h = LinearHash::new(a, m);
            let lhs = (h.eval(x) + h.eval(y)) % m;
            let rhs = h.eval(x.wrapping_add(y));
            let phi = (lhs + m - rhs) % m;
            prop_assert!(phi == 0 || phi == m - 1);
        }
    }
}
