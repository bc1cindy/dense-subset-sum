//! Correlation and summary statistics for validation.

pub fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n.is_multiple_of(2) {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

/// Rank correlation in [-1, 1] (scale-invariant via ranks).
#[must_use]
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let rank_x = ranks(x);
    let rank_y = ranks(y);

    let mean_x: f64 = rank_x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = rank_y.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..n {
        let dx = rank_x[i] - mean_x;
        let dy = rank_y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    let den = (den_x * den_y).sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

/// Pearson correlation in [-1, 1].
#[must_use]
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x * den_y).sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

/// Converts values to 1-based ranks (Spearman input); ties get averaged ranks.
fn ranks(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            result[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_pearson_correlation_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);

        let y_rev: Vec<f64> = y.iter().rev().copied().collect();
        let r2 = pearson_correlation(&x, &y_rev);
        assert!((r2 + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_monotone_is_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![10.0, 100.0, 1000.0, 10000.0];
        assert!((spearman_correlation(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_odd_even() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    fn finite_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-1_000.0..1_000.0_f64, min_len..=max_len)
    }

    fn nondegenerate_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f64>> {
        finite_vec(min_len, max_len).prop_filter("must have variance", |v| {
            v.iter().any(|&x| (x - v[0]).abs() > 1e-9)
        })
    }

    proptest! {
        // ---- pearson invariants ----

        /// PCC(x, x) = 1 for any non-degenerate x.
        #[test]
        fn pearson_self_is_one(x in nondegenerate_vec(2, 30)) {
            let r = pearson_correlation(&x, &x);
            prop_assert!((r - 1.0).abs() < 1e-9, "got {r}");
        }

        /// PCC(x, -x) = -1.
        #[test]
        fn pearson_anticorrelation(x in nondegenerate_vec(2, 30)) {
            let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
            let r = pearson_correlation(&x, &neg_x);
            prop_assert!((r + 1.0).abs() < 1e-9, "got {r}");
        }

        /// PCC is symmetric in its arguments.
        #[test]
        fn pearson_symmetric(
            x in finite_vec(2, 30),
            y in finite_vec(2, 30),
        ) {
            let n = x.len().min(y.len());
            let xs = &x[..n];
            let ys = &y[..n];
            let r1 = pearson_correlation(xs, ys);
            let r2 = pearson_correlation(ys, xs);
            if r1.is_nan() {
                prop_assert!(r2.is_nan());
            } else {
                prop_assert!((r1 - r2).abs() < 1e-12, "{r1} vs {r2}");
            }
        }

        /// PCC ∈ [-1, 1] (or NaN for degenerate input).
        #[test]
        fn pearson_bounded(
            x in finite_vec(2, 30),
            y in finite_vec(2, 30),
        ) {
            let n = x.len().min(y.len());
            let r = pearson_correlation(&x[..n], &y[..n]);
            if !r.is_nan() {
                prop_assert!((-1.0 - 1e-9..=1.0 + 1e-9).contains(&r), "out of range: {r}");
            }
        }

        /// PCC scale-invariant: PCC(x, c*y + d) = sign(c) * PCC(x, y).
        #[test]
        fn pearson_scale_invariant(
            x in nondegenerate_vec(3, 30),
            c in 0.1..100.0_f64,
            d in -100.0..100.0_f64,
        ) {
            let y: Vec<f64> = x.iter().map(|v| c * v + d).collect();
            let r = pearson_correlation(&x, &y);
            prop_assert!((r - 1.0).abs() < 1e-9, "got {r}");
        }

        // ---- spearman invariants ----

        /// Spearman ∈ [-1, 1] (or NaN for degenerate input).
        #[test]
        fn spearman_bounded(
            x in finite_vec(2, 30),
            y in finite_vec(2, 30),
        ) {
            let n = x.len().min(y.len());
            let r = spearman_correlation(&x[..n], &y[..n]);
            if !r.is_nan() {
                prop_assert!((-1.0 - 1e-9..=1.0 + 1e-9).contains(&r), "out of range: {r}");
            }
        }

        /// Spearman is monotone-invariant: applying a strictly-increasing transform
        /// to either side preserves the rank correlation.
        #[test]
        fn spearman_monotone_invariant(
            x in nondegenerate_vec(3, 20),
            y in nondegenerate_vec(3, 20),
        ) {
            let n = x.len().min(y.len());
            let r_orig = spearman_correlation(&x[..n], &y[..n]);
            // cbrt is strictly increasing on R; no overflow on our range.
            let y_mono: Vec<f64> = y[..n].iter().map(|v| v.cbrt()).collect();
            let r_mono = spearman_correlation(&x[..n], &y_mono);
            if !r_orig.is_nan() && !r_mono.is_nan() {
                prop_assert!((r_orig - r_mono).abs() < 1e-9, "{r_orig} vs {r_mono}");
            }
        }

        /// Spearman equals Pearson on rank-transformed inputs.
        #[test]
        fn spearman_equals_pearson_on_ranks(
            x in finite_vec(3, 20),
            y in finite_vec(3, 20),
        ) {
            let n = x.len().min(y.len());
            let s = spearman_correlation(&x[..n], &y[..n]);
            let p = pearson_correlation(&ranks(&x[..n]), &ranks(&y[..n]));
            if !s.is_nan() && !p.is_nan() {
                prop_assert!((s - p).abs() < 1e-9, "spearman={s} pearson_of_ranks={p}");
            }
        }

        // ---- median invariants ----

        /// Median equals the middle of the sorted array (odd length).
        #[test]
        fn median_odd_is_middle(vals in finite_vec(1, 25).prop_filter("odd", |v| !v.len().is_multiple_of(2))) {
            let mut sorted = vals.clone();
            sorted.sort_by(f64::total_cmp);
            let mid = sorted[sorted.len() / 2];
            prop_assert_eq!(median(&vals), mid);
        }

        /// Median equals the midpoint of the two middle elements (even length).
        #[test]
        fn median_even_is_midpoint(vals in finite_vec(2, 24).prop_filter("even", |v| v.len().is_multiple_of(2))) {
            let mut sorted = vals.clone();
            sorted.sort_by(f64::total_cmp);
            let n = sorted.len();
            let expected = f64::midpoint(sorted[n / 2 - 1], sorted[n / 2]);
            prop_assert_eq!(median(&vals), expected);
        }

        /// Median is permutation-invariant.
        #[test]
        fn median_permutation_invariant(vals in finite_vec(1, 20)) {
            let m1 = median(&vals);
            let mut shuffled = vals.clone();
            shuffled.reverse();
            let m2 = median(&shuffled);
            prop_assert_eq!(m1, m2);
        }

        /// Median of a single element is that element.
        #[test]
        fn median_singleton(v in -1_000.0..1_000.0_f64) {
            prop_assert_eq!(median(&[v]), v);
        }
    }
}
