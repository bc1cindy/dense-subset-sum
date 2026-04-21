//! Base types and exhaustive / DP-ground-truth comparison helpers.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::stats::{median, spearman_correlation};
use crate::{dp_w, kappa, log_w_for_e_sat, lookup_w};

/// Ground truth for `w_exact`: exhaustive enumeration or Monte Carlo estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareMode {
    /// Full 2^N enumeration; `w_exact` is the integer count.
    Exhaustive,
    /// Bernoulli(0.5) subset sampling; `w_exact` is the scaled estimate
    /// `count × 2^N / samples_drawn`. `timed_out` is true if the run
    /// stopped before `samples_requested` completed.
    MonteCarlo {
        samples_requested: u64,
        samples_drawn: u64,
        timed_out: bool,
    },
}

#[derive(Debug, Clone)]
pub struct ComparisonRow {
    pub e_target: u64,
    /// Integer count in Exhaustive mode; scaled W estimate in MonteCarlo mode.
    pub w_exact: f64,
    pub w_sasamoto: Option<f64>,
    pub w_lookup: Option<u128>,
    pub w_dp: Option<u128>,
    pub err_sasamoto: Option<f64>,
    pub err_lookup: Option<f64>,
    pub err_dp: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct EstimatorSummary {
    pub name: String,
    pub n_points: usize,
    pub median_error: f64,
    pub max_error: f64,
    pub spearman: f64,
}

#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub label: String,
    pub n: usize,
    pub values: Vec<u64>,
    pub kappa: Option<f64>,
    pub rows: Vec<ComparisonRow>,
    pub sasamoto: EstimatorSummary,
    pub lookup: EstimatorSummary,
    pub dp: EstimatorSummary,
    pub mode: CompareMode,
}

pub fn compare(
    a: &[u64],
    min_w: u64,
    lookup_k: usize,
    dp_max: usize,
    label: &str,
) -> ComparisonReport {
    let n = a.len();
    assert!(
        n <= 25,
        "compare: N={} too large for exhaustive enumeration; use compare_monte_carlo",
        n
    );

    let mut counts: HashMap<u64, u64> = HashMap::new();
    for mask in 0..(1u64 << n) {
        let sum: u64 = (0..n).filter(|&j| mask & (1 << j) != 0).map(|j| a[j]).sum();
        *counts.entry(sum).or_insert(0) += 1;
    }

    build_report_from_counts(
        a,
        &counts,
        1.0,
        min_w,
        lookup_k,
        dp_max,
        label,
        CompareMode::Exhaustive,
    )
}

/// Which estimators are ground truth vs. asymptotic diagnostic for a given (N, W range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareRegime {
    /// N ≤ 20, small W — Lookup/DP exact; Sasamoto out of regime.
    Exact,
    /// N ≥ 50 or median W ≥ 100 — all three in asymptotic regime.
    Asymptotic,
    Intermediate,
}

pub fn classify_regime(report: &ComparisonReport) -> CompareRegime {
    let median_w = {
        let mut ws: Vec<f64> = report.rows.iter().map(|r| r.w_exact).collect();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if ws.is_empty() { 0.0 } else { ws[ws.len() / 2] }
    };
    if report.n <= 20 && median_w < 50.0 {
        CompareRegime::Exact
    } else if report.n >= 50 || median_w >= 100.0 {
        CompareRegime::Asymptotic
    } else {
        CompareRegime::Intermediate
    }
}

/// DP as ground truth — no brute force.
///
/// Returns `Err` on invalid input (empty / all-zero) or when `Σa/gcd`
/// exceeds `dp_max` (caller should raise `--dp-max` or lower L/N).
pub fn compare_dp_ground_truth(
    a: &[u64],
    min_w: u64,
    lookup_k: usize,
    dp_max: usize,
    label: &str,
) -> Result<ComparisonReport, String> {
    let n = a.len();
    let g = crate::gcd_slice(a);
    if g == 0 {
        return Err("compare_dp_ground_truth: empty or all-zero input".into());
    }
    let a_norm: Vec<u64> = a.iter().map(|&v| v / g).collect();
    let sum_max: u64 = a_norm.iter().sum();
    if (sum_max as usize) > dp_max {
        return Err(format!(
            "Σa/gcd = {} exceeds dp_max = {}; raise --dp-max or lower L/N",
            sum_max, dp_max
        ));
    }

    let sz = sum_max as usize + 1;
    let mut dp = vec![0u128; sz];
    dp[0] = 1;
    for &val in &a_norm {
        let v = val as usize;
        for j in (v..sz).rev() {
            dp[j] += dp[j - v];
        }
    }

    let k = kappa(a);
    let mut rows: Vec<ComparisonRow> = Vec::new();
    let mut sas_points: Vec<(f64, f64)> = Vec::new();
    let mut lkp_points: Vec<(f64, f64)> = Vec::new();
    let mut dp_points: Vec<(f64, f64)> = Vec::new();

    // Subsample: Lookup is O(N · Σa) per call; cap to keep runtime bounded.
    const MAX_POINTS: usize = 250;
    let achievable: Vec<usize> = (1..sum_max as usize)
        .filter(|&e_norm| dp[e_norm] >= min_w as u128)
        .collect();
    let stride = achievable.len().div_ceil(MAX_POINTS).max(1);
    let sampled: Vec<usize> = achievable.into_iter().step_by(stride).collect();

    for e_norm in sampled {
        let w = dp[e_norm];
        if w < min_w as u128 {
            continue;
        }
        let e = (e_norm as u64) * g;
        let w_f = w as f64;

        let w_sas = log_w_for_e_sat(a, e).map(|lw| lw.exp());
        let w_lkp = lookup_w(a, e, lookup_k);

        let err_sas = w_sas.map(|ws| (ws - w_f) / w_f);
        let err_lkp = w_lkp.map(|wl| (wl as f64 - w_f) / w_f);

        if let Some(ws) = w_sas {
            sas_points.push((ws, w_f));
        }
        if let Some(wl) = w_lkp {
            lkp_points.push((wl as f64, w_f));
        }
        dp_points.push((w_f, w_f));

        rows.push(ComparisonRow {
            e_target: e,
            w_exact: w as f64,
            w_sasamoto: w_sas,
            w_lookup: w_lkp,
            w_dp: Some(w),
            err_sasamoto: err_sas,
            err_lookup: err_lkp,
            err_dp: Some(0.0),
        });
    }

    let sasamoto = make_summary(
        "sasamoto",
        &sas_points,
        &rows
            .iter()
            .filter_map(|r| r.err_sasamoto)
            .collect::<Vec<_>>(),
    );
    let lookup = make_summary(
        &format!("lookup_k{}", lookup_k),
        &lkp_points,
        &rows.iter().filter_map(|r| r.err_lookup).collect::<Vec<_>>(),
    );
    let dp = make_summary(
        "dp (ground truth)",
        &dp_points,
        &rows.iter().filter_map(|r| r.err_dp).collect::<Vec<_>>(),
    );

    Ok(ComparisonReport {
        label: label.to_string(),
        n,
        values: a.to_vec(),
        kappa: k,
        rows,
        sasamoto,
        lookup,
        dp,
        mode: CompareMode::Exhaustive,
    })
}

pub fn uniform_random_set(n: usize, max_val: u64, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.r#gen_range(1..=max_val)).collect()
}

#[derive(Debug, Clone)]
pub struct BatchRow {
    pub config: String,
    pub n_sets: usize,
    pub avg_sas_median_err: f64,
    pub avg_lkp_median_err: f64,
    pub avg_sas_spearman: f64,
    pub avg_lkp_spearman: f64,
}

pub fn aggregate_reports(reports: &[ComparisonReport], config: &str) -> BatchRow {
    let n = reports.len();
    if n == 0 {
        return BatchRow {
            config: config.to_string(),
            n_sets: 0,
            avg_sas_median_err: f64::NAN,
            avg_lkp_median_err: f64::NAN,
            avg_sas_spearman: f64::NAN,
            avg_lkp_spearman: f64::NAN,
        };
    }

    let avg = |vals: &[f64]| -> f64 {
        let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            f64::NAN
        } else {
            finite.iter().sum::<f64>() / finite.len() as f64
        }
    };

    BatchRow {
        config: config.to_string(),
        n_sets: n,
        avg_sas_median_err: avg(&reports
            .iter()
            .map(|r| r.sasamoto.median_error)
            .collect::<Vec<_>>()),
        avg_lkp_median_err: avg(&reports
            .iter()
            .map(|r| r.lookup.median_error)
            .collect::<Vec<_>>()),
        avg_sas_spearman: avg(&reports
            .iter()
            .map(|r| r.sasamoto.spearman)
            .collect::<Vec<_>>()),
        avg_lkp_spearman: avg(&reports
            .iter()
            .map(|r| r.lookup.spearman)
            .collect::<Vec<_>>()),
    }
}

/// Shared body for `compare` and `compare_monte_carlo`.
///
/// `counts[e]` is the number of subsets observed summing to `e`. The estimated
/// `W(e)` is `counts[e] * scale`: exhaustive passes `scale=1.0` so `w_exact`
/// stays integer-valued; MC passes `2^N / samples_drawn` to extrapolate.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_report_from_counts(
    a: &[u64],
    counts: &HashMap<u64, u64>,
    scale: f64,
    min_count: u64,
    lookup_k: usize,
    dp_max: usize,
    label: &str,
    mode: CompareMode,
) -> ComparisonReport {
    let n = a.len();
    let e_max: u64 = a.iter().sum();
    let k = kappa(a);

    let mut targets: Vec<(u64, u64)> = counts
        .iter()
        .filter(|&(&e, &c)| c >= min_count && e > 0 && e < e_max)
        .map(|(&e, &c)| (e, c))
        .collect();
    targets.sort_by_key(|&(e, _)| e);

    let mut rows: Vec<ComparisonRow> = Vec::new();
    let mut sas_points: Vec<(f64, f64)> = Vec::new();
    let mut lkp_points: Vec<(f64, f64)> = Vec::new();
    let mut dp_points: Vec<(f64, f64)> = Vec::new();

    for &(e, c) in &targets {
        let w_ref = c as f64 * scale;
        let w_sas = log_w_for_e_sat(a, e).map(|lw| lw.exp());
        let w_lkp = lookup_w(a, e, lookup_k);
        let w_dp = dp_w(a, e, dp_max);

        let err_sas = w_sas.map(|ws| (ws - w_ref) / w_ref);
        let err_lkp = w_lkp.map(|wl| (wl as f64 - w_ref) / w_ref);
        let err_dp = w_dp.map(|wd| (wd as f64 - w_ref) / w_ref);

        if let Some(ws) = w_sas {
            sas_points.push((ws, w_ref));
        }
        if let Some(wl) = w_lkp {
            lkp_points.push((wl as f64, w_ref));
        }
        if let Some(wd) = w_dp {
            dp_points.push((wd as f64, w_ref));
        }

        rows.push(ComparisonRow {
            e_target: e,
            w_exact: w_ref,
            w_sasamoto: w_sas,
            w_lookup: w_lkp,
            w_dp,
            err_sasamoto: err_sas,
            err_lookup: err_lkp,
            err_dp,
        });
    }

    let sasamoto = make_summary(
        "sasamoto",
        &sas_points,
        &rows
            .iter()
            .filter_map(|r| r.err_sasamoto)
            .collect::<Vec<_>>(),
    );
    let lookup = make_summary(
        &format!("lookup_k{}", lookup_k),
        &lkp_points,
        &rows.iter().filter_map(|r| r.err_lookup).collect::<Vec<_>>(),
    );
    let dp = make_summary(
        "dp",
        &dp_points,
        &rows.iter().filter_map(|r| r.err_dp).collect::<Vec<_>>(),
    );

    ComparisonReport {
        label: label.to_string(),
        n,
        values: a.to_vec(),
        kappa: k,
        rows,
        sasamoto,
        lookup,
        dp,
        mode,
    }
}

pub(super) fn make_summary(name: &str, points: &[(f64, f64)], errors: &[f64]) -> EstimatorSummary {
    let abs_errors: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    let median_error = median(&abs_errors);
    let max_error = errors.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);
    let spearman = if points.len() >= 3 {
        let x: Vec<f64> = points.iter().map(|p| p.0).collect();
        let y: Vec<f64> = points.iter().map(|p| p.1).collect();
        spearman_correlation(&x, &y)
    } else {
        f64::NAN
    };

    EstimatorSummary {
        name: name.to_string(),
        n_points: points.len(),
        median_error,
        max_error,
        spearman,
    }
}
