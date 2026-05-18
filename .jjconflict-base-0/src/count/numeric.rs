//! Numeric helpers shared across modules.

/// Bisection on a monotone `f`. Caller ensures bracket contains the root.
pub(crate) fn bisect<F: FnMut(f64) -> f64>(
    mut f: F,
    mut lo: f64,
    mut hi: f64,
    target: f64,
    max_iter: u32,
    tol: f64,
) -> f64 {
    let sign_lo = (f(lo) - target).signum();
    for _ in 0..max_iter {
        if hi - lo < tol {
            break;
        }
        let mid = f64::midpoint(lo, hi);
        if (f(mid) - target).signum() == sign_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// gcd of a slice; `None` when slice is empty or all-zero.
pub(crate) fn gcd_slice(vals: &[u64]) -> Option<u64> {
    let g = vals.iter().copied().fold(0, gcd);
    if g == 0 { None } else { Some(g) }
}

fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisect_finds_root_increasing() {
        let r = bisect(|x| x * x - 2.0, 0.0, 2.0, 0.0, 200, 1e-12);
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn bisect_finds_root_decreasing() {
        let r = bisect(|x| -x, -10.0, 10.0, 0.0, 200, 1e-12);
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn gcd_slice_empty_is_none() {
        assert_eq!(gcd_slice(&[]), None);
    }

    #[test]
    fn gcd_slice_all_zero_is_none() {
        assert_eq!(gcd_slice(&[0, 0, 0]), None);
    }

    #[test]
    fn gcd_slice_basic() {
        assert_eq!(gcd_slice(&[12, 18, 30]), Some(6));
        assert_eq!(gcd_slice(&[7]), Some(7));
        assert_eq!(gcd_slice(&[5, 0, 10]), Some(5));
    }
}
