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
}
