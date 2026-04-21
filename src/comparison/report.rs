//! Human-readable and CSV printers for comparison reports and batch summaries.

use super::core::{
    BatchRow, CompareMode, CompareRegime, ComparisonReport, ComparisonRow, EstimatorSummary,
    classify_regime,
};

pub fn print_report(report: &ComparisonReport) {
    let k_str = report
        .kappa
        .map(|k| format!("{:.3}", k))
        .unwrap_or("N/A".into());
    println!("\n=== {} (N={}, κ={}) ===", report.label, report.n, k_str);
    match report.mode {
        CompareMode::Exhaustive => {
            println!(
                "Ground truth: exhaustive enumeration over 2^{} subsets",
                report.n
            );
        }
        CompareMode::MonteCarlo {
            samples_requested,
            samples_drawn,
            timed_out,
        } => {
            let status = if timed_out { "timed out" } else { "completed" };
            println!(
                "Ground truth: Monte Carlo — {}/{} samples ({}); W values are estimates",
                samples_drawn, samples_requested, status
            );
        }
    }

    let regime = classify_regime(report);
    let (min_w, max_w, median_w) = {
        let mut ws: Vec<f64> = report.rows.iter().map(|r| r.w_exact).collect();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if ws.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            (ws[0], *ws.last().unwrap(), ws[ws.len() / 2])
        }
    };
    println!(
        "W range: [{:.0}, {:.0}], median={:.0}  →  Regime: {}",
        min_w,
        max_w,
        median_w,
        match regime {
            CompareRegime::Exact => "EXACT (N≤20, small W) — Lookup/DP are ground truth",
            CompareRegime::Asymptotic =>
                "ASYMPTOTIC (N≥50 or W≥100) — Sasamoto in regime, usable for correlation",
            CompareRegime::Intermediate => "INTERMEDIATE — all estimators informative",
        }
    );
    println!(
        "{:>10}  {:>10}  {:>12}  {:>12}  {:>12}  {:>9}  {:>9}  {:>9}",
        "E", "W_exact", "Sasamoto", "Lookup", "DP", "Sas_err", "Lkp_err", "DP_err"
    );
    println!("{:─<97}", "");

    let show_rows: Vec<&ComparisonRow> = if report.rows.len() > MAX_PRINTED_ROWS {
        let mut v: Vec<&ComparisonRow> = report.rows[..PREVIEW_HEAD].iter().collect();
        v.push(&report.rows[report.rows.len() / 2]);
        v.extend(report.rows[report.rows.len() - PREVIEW_TAIL..].iter());
        v
    } else {
        report.rows.iter().collect()
    };

    for row in &show_rows {
        let sas_str = row
            .w_sasamoto
            .map(|w| format!("{:.1}", w))
            .unwrap_or("N/A".into());
        let lkp_str = row
            .w_lookup
            .map(|w| format!("{}", w))
            .unwrap_or("N/A".into());
        let dp_str = row.w_dp.map(|w| format!("{}", w)).unwrap_or("N/A".into());
        let sas_err = row
            .err_sasamoto
            .map(|e| format!("{:+.1}%", e * 100.0))
            .unwrap_or("".into());
        let lkp_err = row
            .err_lookup
            .map(|e| format!("{:+.1}%", e * 100.0))
            .unwrap_or("".into());
        let dp_err = row
            .err_dp
            .map(|e| format!("{:+.1}%", e * 100.0))
            .unwrap_or("".into());

        println!(
            "{:>10}  {:>10.0}  {:>12}  {:>12}  {:>12}  {:>9}  {:>9}  {:>9}",
            row.e_target, row.w_exact, sas_str, lkp_str, dp_str, sas_err, lkp_err, dp_err
        );
    }

    if report.rows.len() > MAX_PRINTED_ROWS {
        println!(
            "  ... ({} rows total, showing {})",
            report.rows.len(),
            show_rows.len()
        );
    }

    println!();
    match regime {
        CompareRegime::Exact => {
            println!("  [GROUND TRUTH]");
            print_summary_line(&report.lookup);
            print_summary_line(&report.dp);
            println!("  [ASYMPTOTIC DIAGNOSTIC — out of regime at this N, expect deviation]");
            print_summary_line(&report.sasamoto);
        }
        CompareRegime::Asymptotic => {
            println!("  [ASYMPTOTIC REGIME — all three are ground-truth candidates]");
            print_summary_line(&report.sasamoto);
            print_summary_line(&report.lookup);
            print_summary_line(&report.dp);
        }
        CompareRegime::Intermediate => {
            print_summary_line(&report.sasamoto);
            print_summary_line(&report.lookup);
            print_summary_line(&report.dp);
        }
    }
}

pub fn print_report_csv(report: &ComparisonReport) {
    println!("# {}", report.label);
    println!("E,W_exact,W_sasamoto,W_lookup,W_dp,err_sas,err_lkp,err_dp");
    for row in &report.rows {
        println!(
            "{},{:.4},{},{},{},{},{},{}",
            row.e_target,
            row.w_exact,
            row.w_sasamoto
                .map(|w| format!("{:.2}", w))
                .unwrap_or_default(),
            row.w_lookup.map(|w| format!("{}", w)).unwrap_or_default(),
            row.w_dp.map(|w| format!("{}", w)).unwrap_or_default(),
            row.err_sasamoto
                .map(|e| format!("{:.4}", e))
                .unwrap_or_default(),
            row.err_lookup
                .map(|e| format!("{:.4}", e))
                .unwrap_or_default(),
            row.err_dp.map(|e| format!("{:.4}", e)).unwrap_or_default(),
        );
    }
}

pub fn print_batch_summary(rows: &[BatchRow]) {
    println!(
        "\n{:<30}  {:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Config", "Sets", "Sas_med_err", "Lkp_med_err", "Sas_spear", "Lkp_spear"
    );
    println!("{:─<90}", "");
    for row in rows {
        println!(
            "{:<30}  {:>6}  {:>11.1}%  {:>11.1}%  {:>12.4}  {:>12.4}",
            row.config,
            row.n_sets,
            row.avg_sas_median_err * 100.0,
            row.avg_lkp_median_err * 100.0,
            row.avg_sas_spearman,
            row.avg_lkp_spearman,
        );
    }
}

/// Console rendering budget: above this row count the printer switches to a
/// head/median/tail preview (see `PREVIEW_HEAD` / `PREVIEW_TAIL`) instead of
/// dumping every row. Use `print_report_csv` for the full table.
const MAX_PRINTED_ROWS: usize = 30;
const PREVIEW_HEAD: usize = 10;
const PREVIEW_TAIL: usize = 10;

fn print_summary_line(s: &EstimatorSummary) {
    println!(
        "  {:<15} pts={:<4} median_err={:.1}%  max_err={:.1}%  spearman={:.4}",
        s.name,
        s.n_points,
        s.median_error * 100.0,
        s.max_error * 100.0,
        s.spearman,
    );
}
