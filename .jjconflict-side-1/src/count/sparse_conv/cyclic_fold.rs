//! Cyclic ⋆_p at arbitrary `p`: NTT at next-pow-of-2, fold wrap-around mod p.

use super::field::Field;

pub(super) fn cyclic_convolve<F: Field>(a: &[u64], b: &[u64], p: usize) -> Vec<u64> {
    debug_assert!(p >= 1, "output length must be positive");
    debug_assert!(a.len() <= p && b.len() <= p, "inputs must fit in p");
    if a.is_empty() || b.is_empty() || p == 0 {
        return vec![0u64; p];
    }

    let lin_len = a.len() + b.len() - 1;
    let n_pad = lin_len.next_power_of_two().max(1);
    let lin = F::cyclic_convolve(a, b, n_pad);

    let mut out = vec![0u64; p];
    for (j, &v) in lin.iter().take(lin_len).enumerate() {
        let k = j % p;
        out[k] = F::add(out[k], v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::ntt::{self, Goldilocks};
    use super::*;
    use proptest::prelude::*;

    fn naive_cyclic(a: &[u64], b: &[u64], p: usize) -> Vec<u64> {
        let mut out = vec![0u64; p];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                let xp = x % ntt::P;
                let yp = y % ntt::P;
                let prod = Goldilocks::mul(xp, yp);
                let k = (i + j) % p;
                out[k] = Goldilocks::add(out[k], prod);
            }
        }
        out
    }

    #[test]
    fn delta_at_length_five() {
        let mut delta = vec![0u64; 5];
        delta[0] = 1;
        let v = vec![10u64, 20, 30, 40, 50];
        assert_eq!(cyclic_convolve::<Goldilocks>(&v, &delta, 5), v);
    }

    #[test]
    fn no_wrap_when_lin_len_fits_in_p() {
        assert_eq!(
            cyclic_convolve::<Goldilocks>(&[1, 2], &[3, 4], 3),
            vec![3, 10, 8]
        );
    }

    #[test]
    fn wraps_when_lin_len_exceeds_p() {
        assert_eq!(
            cyclic_convolve::<Goldilocks>(&[1, 2], &[3, 4], 2),
            vec![11, 10]
        );
    }

    #[test]
    fn matches_naive_for_prime_lengths() {
        for &p in &[3usize, 5, 7, 11, 13, 17, 31, 53, 97] {
            let a: Vec<u64> = (0..p as u64).map(|i| i * 7 + 3).collect();
            let b: Vec<u64> = (0..p as u64).map(|i| i * 11 + 5).collect();
            assert_eq!(
                cyclic_convolve::<Goldilocks>(&a, &b, p),
                naive_cyclic(&a, &b, p),
                "p={p}"
            );
        }
    }

    proptest! {
        #[test]
        fn prop_matches_naive(
            a in proptest::collection::vec(0u64..1_000_000, 0..16),
            b in proptest::collection::vec(0u64..1_000_000, 0..16),
            p_offset in 0usize..16,
        ) {
            let p = a.len().max(b.len()).max(1) + p_offset;
            prop_assert_eq!(cyclic_convolve::<Goldilocks>(&a, &b, p), naive_cyclic(&a, &b, p));
        }
    }
}
