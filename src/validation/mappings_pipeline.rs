//! Compare W(E) lower bounds against Maurer/Boltzmann CJA mapping cardinality.
//!
//! Central hypothesis: `max log W` over unique sub-txs correlates with
//! `ln(#non_derived_mappings + 1)` across many transactions.

use super::sub_tx_estimates::{SubTxEstimate, estimate_sub_txs};
use crate::mappings;
use crate::stats::{median, pearson_correlation, spearman_correlation};
use crate::{Transaction, is_radix_like_in_base, log_w_signed_adaptive};

#[derive(Debug, Clone)]
pub struct ValidationSummary {
    pub n_transactions: usize,
    /// (name, median_error, spearman)
    pub estimator_stats: Vec<(String, f64, f64)>,
    pub regime_notes: Vec<String>,
}

pub fn validate_estimators(
    test_sets: &[Vec<u64>],
    min_w: u64,
    lookup_k: usize,
    dp_max_table: usize,
) -> ValidationSummary {
    let mut all_sasamoto_errs = Vec::new();
    let mut all_lookup_errs = Vec::new();
    let mut all_dp_errs = Vec::new();
    let mut regime_notes = Vec::new();

    for (i, a) in test_sets.iter().enumerate() {
        let label = format!("set_{}", i);
        let report = crate::comparison::compare(a, min_w, lookup_k, dp_max_table, &label);

        for row in &report.rows {
            if let Some(e) = row.err_sasamoto {
                all_sasamoto_errs.push(e.abs());
            }
            if let Some(e) = row.err_lookup {
                all_lookup_errs.push(e.abs());
            }
            if let Some(e) = row.err_dp {
                all_dp_errs.push(e.abs());
            }
        }

        let n = a.len();
        if n < 10 && report.sasamoto.median_error > 0.2 {
            regime_notes.push(format!(
                "N={}: Sasamoto median error {:.1}% (small N regime)",
                n,
                report.sasamoto.median_error * 100.0
            ));
        }

        if is_radix_like_in_base(a, 2, 1) {
            regime_notes.push(format!("N={}: radix-like values detected", n));
        }
    }

    let stats = vec![
        ("sasamoto".into(), median(&all_sasamoto_errs), 0.0),
        (
            format!("lookup_k{}", lookup_k),
            median(&all_lookup_errs),
            0.0,
        ),
        ("dp".into(), median(&all_dp_errs), 0.0),
    ];

    ValidationSummary {
        n_transactions: test_sets.len(),
        estimator_stats: stats,
        regime_notes,
    }
}

/// How to balance an unbalanced tx before CJA mapping enumeration.
#[derive(Debug, Clone, Copy)]
pub enum FeeHandling {
    /// Append `fee` as an extra output (canonical Maurer/Boltzmann).
    PhantomOutput,
    /// Signed multiset model: inputs +v, outputs −u, valid sub-tx sums to `fee`.
    SignedMultiset,
}

#[derive(Debug, Clone)]
pub struct MappingComparison {
    pub label: String,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub total_coins: usize,
    pub fee: u64,
    pub fee_handling: FeeHandling,
    pub n_mappings: usize,
    pub n_non_derived: usize,
    pub sub_tx_estimates: Vec<SubTxEstimate>,
    pub max_log_w_sasamoto: f64,
    pub max_log_w_lookup: f64,
    /// Bilateral max: `NEG_INFINITY` if no sub-tx has finite combined value.
    pub max_log_w_combined: f64,
    /// Populated only under `SignedMultiset`; target = fee, computed on original tx.
    pub log_w_signed: Option<f64>,
}

pub fn compare_w_vs_mappings(
    tx: &Transaction,
    label: &str,
    lookup_k: usize,
    max_coins: usize,
) -> Option<MappingComparison> {
    compare_w_vs_mappings_with(tx, label, lookup_k, max_coins, FeeHandling::PhantomOutput)
}

/// Returns `None` when total balanced coins exceed `max_coins` or the tx has a negative fee.
pub fn compare_w_vs_mappings_with(
    tx: &Transaction,
    label: &str,
    lookup_k: usize,
    max_coins: usize,
    fee_handling: FeeHandling,
) -> Option<MappingComparison> {
    let fee = tx.fee();

    let balanced_tx = if fee > 0 {
        let mut outputs = tx.outputs.clone();
        outputs.push(fee);
        Transaction::new(tx.inputs.clone(), outputs)
    } else {
        tx.clone()
    };

    let total_coins = balanced_tx.inputs.len() + balanced_tx.outputs.len();
    if total_coins > max_coins {
        return None;
    }

    let all_mappings = mappings::enumerate_mappings(&balanced_tx);
    let non_derived = mappings::non_derived_mappings(&all_mappings);

    let estimates = estimate_sub_txs(&balanced_tx, lookup_k);

    let max_log_w_sas = estimates
        .iter()
        .filter_map(|e| e.log_w_sasamoto)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_log_w_lkp = estimates
        .iter()
        .filter_map(|e| e.log_w_lookup)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_log_w_combined = estimates
        .iter()
        .filter_map(|e| e.log_w_combined)
        .fold(f64::NEG_INFINITY, f64::max);

    let log_w_signed = if matches!(fee_handling, FeeHandling::SignedMultiset) {
        log_w_signed_adaptive(&tx.inputs, &tx.outputs, fee as i64, lookup_k)
    } else {
        None
    };

    Some(MappingComparison {
        label: label.to_string(),
        n_inputs: tx.inputs.len(),
        n_outputs: tx.outputs.len(),
        total_coins,
        fee,
        fee_handling,
        n_mappings: all_mappings.len(),
        n_non_derived: non_derived.len(),
        sub_tx_estimates: estimates,
        max_log_w_sasamoto: max_log_w_sas,
        max_log_w_lookup: max_log_w_lkp,
        max_log_w_combined,
        log_w_signed,
    })
}

pub fn print_mapping_comparison(mc: &MappingComparison) {
    println!(
        "\n=== {} ({}in/{}out, {} coins, fee={}) ===",
        mc.label, mc.n_inputs, mc.n_outputs, mc.total_coins, mc.fee
    );
    println!(
        "  CJA mappings: {} total, {} non-derived",
        mc.n_mappings, mc.n_non_derived
    );
    println!(
        "  max log_w: sasamoto={:.2}, lookup_in={:.2}, combined(in+out)={:.2}",
        mc.max_log_w_sasamoto, mc.max_log_w_lookup, mc.max_log_w_combined
    );
    if let Some(sw) = mc.log_w_signed {
        println!(
            "  signed log_w (target=fee): {:.2}  ({:.2} log₂ — signed multiset)",
            sw,
            sw / 2.0_f64.ln()
        );
    }

    if !mc.sub_tx_estimates.is_empty() {
        let shown = mc.sub_tx_estimates.len();
        println!("  Unique sub-transactions: {}", shown);
        println!(
            "  {:>3} {:>4} {:>4} {:>5} {:>9} {:>12} {:>9} {:>9} {:>9} {:>9} {:>4}",
            "idx",
            "in",
            "out",
            "count",
            "ln(cnt)",
            "balance",
            "lw_sas",
            "lw_in",
            "lw_out",
            "lw_comb",
            "rel",
        );
        for (i, est) in mc.sub_tx_estimates.iter().enumerate() {
            println!(
                "  {:>3} {:>4} {:>4} {:>5} {:>9.2} {:>12} {:>9} {:>9} {:>9} {:>9} {:>4}",
                i,
                est.inputs.len(),
                est.outputs.len(),
                est.count,
                (est.count as f64).ln(),
                est.balance,
                est.log_w_sasamoto
                    .map_or("N/A".into(), |v| format!("{:.2}", v)),
                est.log_w_lookup
                    .map_or("N/A".into(), |v| format!("{:.2}", v)),
                est.log_w_lookup_outputs
                    .map_or("N/A".into(), |v| format!("{:.2}", v)),
                est.log_w_combined
                    .map_or("N/A".into(), |v| format!("{:.2}", v)),
                if est.sasamoto_reliable { "ok" } else { "!" },
            );
        }
    }
}

/// Central hypothesis: `max_log_w` per tx correlates with `ln(n_non_derived + 1)` across many txs.
#[derive(Debug, Clone)]
pub struct MappingCorrelation {
    pub n_transactions: usize,
    pub spearman_sasamoto: f64,
    pub spearman_lookup: f64,
    pub pearson_sasamoto: f64,
    pub pearson_lookup: f64,
    /// `max_log_w_lookup / ln(n_non_derived + 1)`; ≈1 means lookup is calibrated to mapping entropy.
    pub median_ratio_lookup: f64,
}

/// Skips txs where `n_non_derived == 0` or `max_log_w_lookup` is non-finite.
pub fn correlate_w_vs_mappings(comparisons: &[MappingComparison]) -> Option<MappingCorrelation> {
    let mut log_n: Vec<f64> = Vec::new();
    let mut max_sas: Vec<f64> = Vec::new();
    let mut max_lkp: Vec<f64> = Vec::new();

    for mc in comparisons {
        if mc.n_non_derived == 0 {
            continue;
        }
        if !mc.max_log_w_lookup.is_finite() {
            continue;
        }
        log_n.push(((mc.n_non_derived + 1) as f64).ln());
        let sas = if mc.max_log_w_sasamoto.is_finite() {
            mc.max_log_w_sasamoto
        } else {
            0.0
        };
        max_sas.push(sas);
        max_lkp.push(mc.max_log_w_lookup);
    }

    if log_n.len() < 3 {
        return None;
    }

    let ratios: Vec<f64> = log_n
        .iter()
        .zip(max_lkp.iter())
        .filter(|(ln, _)| **ln > 0.0)
        .map(|(ln, lw)| lw / ln)
        .collect();
    let median_ratio = if ratios.is_empty() {
        f64::NAN
    } else {
        median(&ratios)
    };

    Some(MappingCorrelation {
        n_transactions: log_n.len(),
        spearman_sasamoto: spearman_correlation(&max_sas, &log_n),
        spearman_lookup: spearman_correlation(&max_lkp, &log_n),
        pearson_sasamoto: pearson_correlation(&max_sas, &log_n),
        pearson_lookup: pearson_correlation(&max_lkp, &log_n),
        median_ratio_lookup: median_ratio,
    })
}

pub fn print_mapping_correlation(corr: &MappingCorrelation) {
    println!("\n=== W(E) vs CJA mapping cardinality correlation ===");
    println!("  N transactions: {}", corr.n_transactions);
    println!(
        "  Spearman ρ (rank):    sasamoto={:.4}  lookup={:.4}",
        corr.spearman_sasamoto, corr.spearman_lookup
    );
    println!(
        "  Pearson r  (linear):  sasamoto={:.4}  lookup={:.4}",
        corr.pearson_sasamoto, corr.pearson_lookup
    );
    println!(
        "  Median (log_w_lookup / ln(#non_der+1)): {:.3}",
        corr.median_ratio_lookup
    );
    println!("  (ratio ≈ 1 means lookup is calibrated to mapping entropy)");
}

pub fn print_mapping_summary(comparisons: &[MappingComparison]) {
    println!(
        "\n{:<30} {:>4} {:>4} {:>6} {:>7} {:>7} {:>10} {:>10}",
        "Label", "In", "Out", "Coins", "#Map", "#NonDer", "max_sas", "max_lkp"
    );
    println!("{:─<88}", "");
    for mc in comparisons {
        println!(
            "{:<30} {:>4} {:>4} {:>6} {:>7} {:>7} {:>10.2} {:>10.2}",
            mc.label,
            mc.n_inputs,
            mc.n_outputs,
            mc.total_coins,
            mc.n_mappings,
            mc.n_non_derived,
            mc.max_log_w_sasamoto,
            mc.max_log_w_lookup
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    #[test]
    fn test_validate_estimators_summary() {
        let test_sets: Vec<Vec<u64>> = vec![(1..=16).collect(), (5..=20).collect()];
        let summary = validate_estimators(&test_sets, 50, 8, 1_000_000);

        assert_eq!(summary.n_transactions, 2);
        assert_eq!(summary.estimator_stats.len(), 3);

        for (name, median_err, _) in &summary.estimator_stats {
            eprintln!("{}: median_error={:.1}%", name, median_err * 100.0);
        }
    }

    #[test]
    fn test_correlation_monotone_with_n_non_derived() {
        let rows: Vec<MappingComparison> = (1..=8)
            .map(|i| {
                let log_w = 0.9 * (((i + 1) as f64).ln());
                MappingComparison {
                    label: format!("synth_{}", i),
                    n_inputs: 10,
                    n_outputs: 5,
                    total_coins: 15,
                    fee: 0,
                    fee_handling: FeeHandling::PhantomOutput,
                    n_mappings: (i * 3) as usize,
                    n_non_derived: i as usize,
                    sub_tx_estimates: Vec::new(),
                    max_log_w_sasamoto: log_w - 0.1,
                    max_log_w_lookup: log_w,
                    max_log_w_combined: log_w,
                    log_w_signed: None,
                }
            })
            .collect();

        let corr = correlate_w_vs_mappings(&rows).expect("correlation");
        assert_eq!(corr.n_transactions, 8);
        assert!(
            corr.spearman_lookup > 0.99,
            "expect ρ≈1 for monotone data, got {}",
            corr.spearman_lookup
        );
        assert!(
            corr.pearson_lookup > 0.99,
            "expect Pearson≈1 for linear data in ln(n+1), got {}",
            corr.pearson_lookup
        );
        assert!(
            (corr.median_ratio_lookup - 0.9).abs() < 1e-9,
            "expect slope 0.9, got {}",
            corr.median_ratio_lookup
        );
    }

    #[test]
    fn test_correlation_filters_zero_non_derived() {
        let rows: Vec<MappingComparison> = vec![
            MappingComparison {
                label: "only_derived".into(),
                n_inputs: 2,
                n_outputs: 2,
                total_coins: 4,
                fee: 0,
                fee_handling: FeeHandling::PhantomOutput,
                n_mappings: 1,
                n_non_derived: 0,
                sub_tx_estimates: Vec::new(),
                max_log_w_sasamoto: 5.0,
                max_log_w_lookup: 5.0,
                max_log_w_combined: 5.0,
                log_w_signed: None,
            },
            MappingComparison {
                label: "one".into(),
                n_inputs: 3,
                n_outputs: 3,
                total_coins: 6,
                fee: 0,
                fee_handling: FeeHandling::PhantomOutput,
                n_mappings: 2,
                n_non_derived: 1,
                sub_tx_estimates: Vec::new(),
                max_log_w_sasamoto: 2.0,
                max_log_w_lookup: 2.0,
                max_log_w_combined: 2.0,
                log_w_signed: None,
            },
            MappingComparison {
                label: "two".into(),
                n_inputs: 4,
                n_outputs: 4,
                total_coins: 8,
                fee: 0,
                fee_handling: FeeHandling::PhantomOutput,
                n_mappings: 4,
                n_non_derived: 3,
                sub_tx_estimates: Vec::new(),
                max_log_w_sasamoto: 3.0,
                max_log_w_lookup: 3.0,
                max_log_w_combined: 3.0,
                log_w_signed: None,
            },
            MappingComparison {
                label: "three".into(),
                n_inputs: 5,
                n_outputs: 5,
                total_coins: 10,
                fee: 0,
                fee_handling: FeeHandling::PhantomOutput,
                n_mappings: 6,
                n_non_derived: 5,
                sub_tx_estimates: Vec::new(),
                max_log_w_sasamoto: 4.0,
                max_log_w_lookup: 4.0,
                max_log_w_combined: 4.0,
                log_w_signed: None,
            },
        ];
        let corr = correlate_w_vs_mappings(&rows).expect("correlation");
        assert_eq!(corr.n_transactions, 3);
    }

    #[test]
    fn test_mapping_comparison_populates_max_combined() {
        let tx = fixtures::equal_denominations();
        let mc = compare_w_vs_mappings(&tx, "eqdenom", 4, 16).expect("enumerable");
        assert!(
            mc.max_log_w_combined.is_finite(),
            "max_log_w_combined must be finite when both sides have ambiguity"
        );
        assert!(
            mc.max_log_w_combined >= mc.max_log_w_lookup - 1e-9,
            "combined ({}) must be ≥ input-side lookup ({})",
            mc.max_log_w_combined,
            mc.max_log_w_lookup
        );
    }
}
