//! Prime field for NTT. Impls supply P, G, `TWO_ADIC_ORDER`, add/sub/mul.

/// Finite-field arithmetic backing the NTT in [`super::ntt`].
///
/// # Safety contract for impls
///
/// - `P` must be prime.
/// - `P - 1` must be divisible by `2^TWO_ADIC_ORDER` so NTTs of length up to
///   `2^TWO_ADIC_ORDER` exist (the multiplicative subgroup contains a primitive
///   `2^TWO_ADIC_ORDER`-th root of unity).
/// - `G` must be a primitive root mod `P`: its multiplicative order in
///   `(Z/PZ)*` must equal `P - 1`. The NTT derives all twiddle roots from
///   `G^((P-1)/n)` for `n` a power of two, so a non-primitive `G` produces
///   wrong outputs without panicking.
/// - `add`, `sub`, `mul` must implement modular arithmetic in `Z/PZ` exactly,
///   for any `u64` inputs (callers may pass non-reduced values).
///
/// Violating any of these silently corrupts the NTT output. There is no
/// runtime check; impls are responsible for correctness.
pub trait Field: 'static + Copy {
    const P: u64;
    const G: u64;
    const TWO_ADIC_ORDER: u32;

    fn add(a: u64, b: u64) -> u64;
    fn sub(a: u64, b: u64) -> u64;
    fn mul(a: u64, b: u64) -> u64;

    #[must_use]
    fn pow(mut base: u64, mut exp: u64) -> u64 {
        let mut result = 1u64;
        base %= Self::P;
        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mul(result, base);
            }
            base = Self::mul(base, base);
            exp >>= 1;
        }
        result
    }

    #[must_use]
    fn inv(a: u64) -> u64 {
        Self::pow(a, Self::P - 2)
    }

    fn transform(a: &mut [u64], invert: bool) {
        let n = a.len();
        debug_assert!(n.is_power_of_two(), "NTT length must be a power of two");
        debug_assert!(
            n.trailing_zeros() <= Self::TWO_ADIC_ORDER,
            "NTT length exceeds 2-adic order of P"
        );

        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                a.swap(i, j);
            }
        }

        let mut len = 2;
        let mut twiddles = Vec::with_capacity(n / 2);
        while len <= n {
            let root = Self::pow(Self::G, (Self::P - 1) / len as u64);
            let w_step = if invert { Self::inv(root) } else { root };
            let half = len / 2;
            twiddles.clear();
            let mut w = 1u64;
            for _ in 0..half {
                twiddles.push(w);
                w = Self::mul(w, w_step);
            }
            let mut i = 0;
            while i < n {
                for k in 0..half {
                    let u = a[i + k];
                    let v = Self::mul(a[i + k + half], twiddles[k]);
                    a[i + k] = Self::add(u, v);
                    a[i + k + half] = Self::sub(u, v);
                }
                i += len;
            }
            len <<= 1;
        }

        if invert {
            let n_inv = Self::inv(n as u64);
            for x in a.iter_mut() {
                *x = Self::mul(*x, n_inv);
            }
        }
    }

    /// `m` must be a power of two.
    #[must_use]
    fn cyclic_convolve(a: &[u64], b: &[u64], m: usize) -> Vec<u64> {
        debug_assert!(m.is_power_of_two(), "m must be a power of two");
        debug_assert!(a.len() <= m && b.len() <= m, "inputs must fit in m");
        let mut fa = vec![0u64; m];
        let mut fb = vec![0u64; m];
        for (i, &x) in a.iter().enumerate() {
            fa[i] = x % Self::P;
        }
        for (i, &x) in b.iter().enumerate() {
            fb[i] = x % Self::P;
        }
        Self::transform(&mut fa, false);
        Self::transform(&mut fb, false);
        for i in 0..m {
            fa[i] = Self::mul(fa[i], fb[i]);
        }
        Self::transform(&mut fa, true);
        fa
    }
}
