//! Dense region scan: find E maximizing W(E) and a safe midpoint within the plateau.

use crate::sasamoto::log_w_for_e;

#[derive(Debug, Clone)]
pub struct DenseRegion {
    pub e_peak: f64,
    pub log_w_peak: f64,
    pub w_peak: f64,
    pub e_lo: f64,
    pub e_hi: f64,
    pub e_safe: f64,
    pub log_w_safe: f64,
}

/// Scan E in `n_steps` uniform points across (0, Σaⱼ).
fn scan(a: &[f64], min_log_w: f64, n_steps: usize) -> Vec<(f64, f64)> {
    let e_max: f64 = a.iter().sum();
    let mut pts: Vec<(f64, f64)> = (1..n_steps)
        .filter_map(|i| {
            let e = e_max * i as f64 / n_steps as f64;
            let lw = log_w_for_e(a, e)?;
            if lw >= min_log_w { Some((e, lw)) } else { None }
        })
        .collect();
    pts.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("log_w_for_e never returns NaN")
    });
    pts
}

/// Safe point is the center of [e_lo, e_hi] — maximally far from threshold edges.
pub fn find_dense_region(a: &[f64], min_log_w: f64, n_steps: usize) -> Option<DenseRegion> {
    let pts = scan(a, min_log_w, n_steps);
    let &(e_peak, log_w_peak) = pts.first()?;

    let (mut e_lo, mut e_hi) = (f64::MAX, f64::MIN);
    for &(e, _) in &pts {
        if e < e_lo {
            e_lo = e;
        }
        if e > e_hi {
            e_hi = e;
        }
    }

    let e_safe = (e_lo + e_hi) / 2.0;
    let log_w_safe = log_w_for_e(a, e_safe)?;

    Some(DenseRegion {
        e_peak,
        log_w_peak,
        w_peak: if log_w_peak < 700.0 {
            log_w_peak.exp()
        } else {
            f64::INFINITY
        },
        e_lo,
        e_hi,
        e_safe,
        log_w_safe,
    })
}
