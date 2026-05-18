//! NTT over the Goldilocks prime p = 2^64 - 2^32 + 1.

use super::field::Field;

#[derive(Copy, Clone, Debug)]
pub struct Goldilocks;

pub(super) const P: u64 = 0xFFFF_FFFF_0000_0001;
/// 2^32 - 1 ≡ 2^64 (mod P); used by reduction in `mul`.
const EPSILON: u64 = 0xFFFF_FFFF;

impl Field for Goldilocks {
    const P: u64 = P;
    const G: u64 = 7;
    const TWO_ADIC_ORDER: u32 = 32;

    #[inline]
    fn add(a: u64, b: u64) -> u64 {
        let (s, overflow) = a.overflowing_add(b);
        if overflow || s >= P {
            s.wrapping_sub(P)
        } else {
            s
        }
    }

    #[inline]
    fn sub(a: u64, b: u64) -> u64 {
        let (d, borrow) = a.overflowing_sub(b);
        if borrow { d.wrapping_add(P) } else { d }
    }

    /// Reduction via 2^64 ≡ EPSILON, 2^96 ≡ -1; avoids `% u128`.
    #[inline]
    fn mul(a: u64, b: u64) -> u64 {
        let prod = u128::from(a) * u128::from(b);
        // Intentional bit-level split: low/high 64 bits of u128 product.
        #[allow(clippy::cast_possible_truncation)]
        let x_lo = prod as u64;
        #[allow(clippy::cast_possible_truncation)]
        let x_hi = (prod >> 64) as u64;
        let x_hi_hi = x_hi >> 32;
        let x_hi_lo = x_hi & EPSILON;

        let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
        let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

        let term = x_hi_lo * EPSILON;
        let (t1, carry) = t0.overflowing_add(term);
        let t1 = if carry {
            let (t1_add, carry2) = t1.overflowing_add(EPSILON);
            if carry2 {
                t1_add.wrapping_add(EPSILON)
            } else {
                t1_add
            }
        } else {
            t1
        };

        if t1 >= P { t1 - P } else { t1 }
    }
}

#[cfg(test)]
fn pow(base: u64, exp: u64) -> u64 {
    Goldilocks::pow(base, exp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn naive_cyclic(a: &[u64], b: &[u64], m: usize) -> Vec<u64> {
        let p128 = u128::from(P);
        let mut out = vec![0u128; m];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                let prod = (u128::from(x) % p128) * (u128::from(y) % p128) % p128;
                out[(i + j) % m] = (out[(i + j) % m] + prod) % p128;
            }
        }
        out.into_iter()
            .map(|x| u64::try_from(x).expect("x < P fits u64"))
            .collect()
    }

    fn mul_reference(a: u64, b: u64) -> u64 {
        u64::try_from((u128::from(a) * u128::from(b)) % u128::from(P)).expect("(_ % P) fits u64")
    }

    #[test]
    fn pow_zero_is_one() {
        assert_eq!(pow(123, 0), 1);
        assert_eq!(pow(0, 0), 1);
    }

    #[test]
    fn fermat_little() {
        for x in [2u64, 7, 123_456_789, P - 1] {
            assert_eq!(pow(x, P - 1), 1);
        }
    }

    #[test]
    fn inv_round_trip() {
        for x in [1u64, 2, 3, 7, 999_999_999, P - 1] {
            assert_eq!(Goldilocks::mul(x, Goldilocks::inv(x)), 1);
        }
    }

    #[test]
    fn add_sub_handle_wrap() {
        let big = P - 1;
        assert_eq!(Goldilocks::add(big, big), P - 2);
        assert_eq!(Goldilocks::sub(0, 2), P - 2);
        assert_eq!(Goldilocks::add(big, 1), 0);
        assert_eq!(Goldilocks::sub(0, big), 1);
    }

    #[test]
    fn transform_round_trip() {
        let mut a: Vec<u64> = (1..=8).collect();
        let original = a.clone();
        Goldilocks::transform(&mut a, false);
        Goldilocks::transform(&mut a, true);
        assert_eq!(a, original);
    }

    #[test]
    fn delta_is_convolution_identity() {
        let a: Vec<u64> = (1..=8).collect();
        let mut delta = vec![0u64; 8];
        delta[0] = 1;
        assert_eq!(Goldilocks::cyclic_convolve(&a, &delta, 8), a);
    }

    #[test]
    fn size_one_is_pointwise() {
        assert_eq!(Goldilocks::cyclic_convolve(&[5], &[7], 1), vec![35]);
    }

    #[test]
    fn smaller_input_zero_padded() {
        let m = 8;
        let a = vec![1u64, 2];
        let b = vec![3u64, 4, 5];
        assert_eq!(
            Goldilocks::cyclic_convolve(&a, &b, m),
            naive_cyclic(&a, &b, m)
        );
    }

    #[test]
    fn cyclic_commutative() {
        let m = 16;
        let a: Vec<u64> = (0..m as u64).map(|i| i * i).collect();
        let b: Vec<u64> = (0..m as u64).map(|i| i * 7 + 3).collect();
        assert_eq!(
            Goldilocks::cyclic_convolve(&a, &b, m),
            Goldilocks::cyclic_convolve(&b, &a, m)
        );
    }

    #[test]
    fn cyclic_matches_naive_size_64() {
        let m = 64;
        let a: Vec<u64> = (0..m as u64).map(|i| (i * 31 + 17) % P).collect();
        let b: Vec<u64> = (0..m as u64).map(|i| (i * 71 + 5) % P).collect();
        assert_eq!(
            Goldilocks::cyclic_convolve(&a, &b, m),
            naive_cyclic(&a, &b, m)
        );
    }

    #[test]
    fn mul_matches_reference_corner_cases() {
        let cases = [
            (0u64, 0),
            (1, 1),
            (P - 1, P - 1),
            (P - 1, 1),
            (1, P - 1),
            (1u64 << 32, 1u64 << 32),
            (1u64 << 63, 1u64 << 63),
            (1u64 << 63, 2),
            (2, 1u64 << 63),
            (EPSILON, EPSILON),
            ((P - 1) / 2, 2),
        ];
        for &(a, b) in &cases {
            assert_eq!(Goldilocks::mul(a, b), mul_reference(a, b), "a={a} b={b}");
        }
    }

    proptest! {
        #[test]
        fn prop_cyclic_matches_naive(
            a in proptest::collection::vec(0u64..1_000_000, 0..16),
            b in proptest::collection::vec(0u64..1_000_000, 0..16),
            log_m in 0u32..=4u32,
        ) {
            let m = 1usize << log_m;
            prop_assume!(a.len() <= m && b.len() <= m);
            prop_assert_eq!(Goldilocks::cyclic_convolve(&a, &b, m), naive_cyclic(&a, &b, m));
        }

        #[test]
        fn prop_round_trip(
            xs in proptest::collection::vec(0u64..P, 1..=64),
            log_m in 0u32..=6u32,
        ) {
            let m = 1usize << log_m;
            prop_assume!(xs.len() <= m);
            let mut padded = vec![0u64; m];
            padded[..xs.len()].copy_from_slice(&xs);
            let original = padded.clone();
            Goldilocks::transform(&mut padded, false);
            Goldilocks::transform(&mut padded, true);
            prop_assert_eq!(padded, original);
        }

        #[test]
        fn prop_mul_matches_reference(a in 0u64..P, b in 0u64..P) {
            prop_assert_eq!(Goldilocks::mul(a, b), mul_reference(a, b));
        }
    }
}
