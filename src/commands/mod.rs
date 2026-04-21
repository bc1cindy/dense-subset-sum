//! CLI subcommand implementations. One `cmd_*` fn per `Command` variant.
//!
//! Split across three files by domain:
//! - `analysis`: per-tx reports, per-coin measurements, estimator correlation, splits.
//! - `compare`: W(E)-vs-ground-truth + W-vs-mappings benchmarks.
//! - `density`: κ/κ_c regime tooling and subset-size sweeps.
//!
//! Shared parsing / tx-resolution helpers live in this module and are
//! `pub(super)` for the three submodules.

mod analysis;
mod compare;
mod density;

pub(crate) use analysis::{
    cmd_analyze_tx, cmd_coin_measures, cmd_compare_augmented, cmd_correlate_estimators,
    cmd_estimate, cmd_full_report, cmd_measure, cmd_suggest_split,
};
pub(crate) use compare::{
    cmd_compare, cmd_compare_fixtures, cmd_compare_random, cmd_compare_synthetic,
    cmd_compare_wasabi2, cmd_validate,
};
pub(crate) use density::{
    cmd_dense_boundary, cmd_density, cmd_density_scan, cmd_empirical_nc, cmd_kappa,
    cmd_subset_density,
};

use dense_subset_sum::{SignedMethod, Transaction, fixtures, stats, validation};

pub(super) fn parse_signed_method(s: &str) -> SignedMethod {
    match s {
        "sasamoto" => SignedMethod::Sasamoto,
        "lookup" => SignedMethod::Lookup,
        other => {
            eprintln!(
                "invalid --signed-method value: {:?} (expected \"sasamoto\" or \"lookup\")",
                other
            );
            std::process::exit(1);
        }
    }
}

pub(super) fn parse_tx(inputs_str: &str, outputs_str: &str) -> Transaction {
    Transaction::new(parse_values(inputs_str), parse_values(outputs_str))
}

pub(super) fn parse_values(s: &str) -> Vec<u64> {
    s.split(',')
        .map(|v| v.trim().parse::<u64>().expect("invalid integer value"))
        .collect()
}

pub(super) fn parse_values_f64(s: &str) -> Vec<f64> {
    s.split(',')
        .map(|v| v.trim().parse::<f64>().expect("invalid numeric value"))
        .collect()
}

pub(super) fn spearman_opt(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    let r = stats::spearman_correlation(x, y);
    if r.is_nan() { None } else { Some(r) }
}

#[derive(serde::Deserialize)]
struct TxJson {
    #[serde(default)]
    label: Option<String>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
}

fn load_tx_json(path: &std::path::Path) -> (String, Transaction) {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error reading {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let parsed: TxJson = serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("error parsing {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let label = parsed.label.unwrap_or_else(|| {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "json_tx".into())
    });
    (label, Transaction::new(parsed.inputs, parsed.outputs))
}

pub(crate) struct TxSpec<'a> {
    pub label: Option<&'a str>,
    pub json: Option<&'a std::path::Path>,
    pub inputs_str: &'a str,
    pub outputs_str: &'a str,
}

pub(super) fn resolve_tx(spec: &TxSpec<'_>) -> (String, Transaction) {
    if let Some(path) = spec.json {
        return load_tx_json(path);
    }
    match spec.label {
        Some(lbl) => {
            let all: Vec<(&'static str, Transaction)> = fixtures::all_wasabi2_false_cjtxs()
                .into_iter()
                .chain(fixtures::all_wasabi2_positive_cjtxs())
                .collect();
            match all.into_iter().find(|(l, _)| *l == lbl) {
                Some((l, tx)) => (l.to_string(), tx),
                None => {
                    eprintln!("unknown tx-label: {}", lbl);
                    eprintln!("available labels: (negatives)");
                    for (l, _) in fixtures::all_wasabi2_false_cjtxs() {
                        eprintln!("  {}", l);
                    }
                    eprintln!("available labels: (positives)");
                    for (l, _) in fixtures::all_wasabi2_positive_cjtxs() {
                        eprintln!("  {}", l);
                    }
                    std::process::exit(1);
                }
            }
        }
        None => {
            if spec.inputs_str.is_empty() || spec.outputs_str.is_empty() {
                eprintln!(
                    "must provide one of: --tx-json <path>, --tx-label <label>, or both --inputs and --outputs"
                );
                std::process::exit(1);
            }
            (
                "user_input".to_string(),
                parse_tx(spec.inputs_str, spec.outputs_str),
            )
        }
    }
}

pub(super) fn resolve_values(
    tx_label: Option<&str>,
    tx_json: Option<&std::path::Path>,
    values_str: &str,
    all_coins: bool,
) -> (String, Vec<u64>) {
    let extract = |tx: Transaction| -> Vec<u64> {
        if all_coins {
            let mut v = tx.inputs;
            v.extend(tx.outputs);
            v
        } else {
            tx.inputs
        }
    };
    if let Some(path) = tx_json {
        let (l, tx) = load_tx_json(path);
        return (l, extract(tx));
    }
    if let Some(lbl) = tx_label {
        let all: Vec<(&'static str, Transaction)> = fixtures::all_wasabi2_false_cjtxs()
            .into_iter()
            .chain(fixtures::all_wasabi2_positive_cjtxs())
            .collect();
        return match all.into_iter().find(|(l, _)| *l == lbl) {
            Some((l, tx)) => (l.to_string(), extract(tx)),
            None => {
                eprintln!("unknown tx-label: {}", lbl);
                std::process::exit(1);
            }
        };
    }
    if values_str.is_empty() {
        eprintln!("must provide one of: --tx-json, --tx-label, or --values");
        std::process::exit(1);
    }
    ("user_input".to_string(), parse_values(values_str))
}

pub(super) fn fmt_log_w(v: Option<f64>, ln2: f64) -> String {
    match v {
        Some(x) if x.is_finite() => format!("log₂={:.4}  W≈{:.3e}", x / ln2, x.exp()),
        Some(x) if x == f64::NEG_INFINITY => "W = 0 (unreachable)".to_string(),
        Some(_) => "NaN".to_string(),
        None => "—".to_string(),
    }
}

pub(super) fn per_coin_summary(tag: &str, measurements: &[validation::CoinMeasurement], ln2: f64) {
    let vals: Vec<f64> = measurements.iter().filter_map(|c| c.log_w_signed).collect();
    if vals.is_empty() {
        println!("  {:>9}: no finite per-coin measurements", tag);
        return;
    }
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    println!(
        "  {:>9}: coins={:>3}/{:>3}  max log₂W={:6.2}  mean={:5.2}  min={:5.2}",
        tag,
        vals.len(),
        measurements.len(),
        max / ln2,
        mean / ln2,
        min / ln2
    );
}
