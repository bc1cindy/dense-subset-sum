//! Per-tx full reports, per-coin measurements, estimator correlation, and split suggestions.

use dense_subset_sum::{
    EmpiricalRegime, brute_force_w, change_split, estimator, fixtures, is_distinguished, kappa,
    log_lookup_w, log_w_for_e_sat, mappings, validation,
};

fn fmt_regime(r: Option<EmpiricalRegime>) -> &'static str {
    match r {
        Some(EmpiricalRegime::EqualAmount) => "equal-amount",
        Some(EmpiricalRegime::RadixGeometric) => "radix-geometric",
        Some(EmpiricalRegime::Arithmetic) => "arithmetic",
        Some(EmpiricalRegime::PathologicalBatch) => "pathological-batch",
        None => "—",
    }
}

use super::{
    Transaction, TxSpec, fmt_log_w, parse_signed_method, parse_tx, parse_values, per_coin_summary,
    resolve_tx, spearman_opt,
};

pub(crate) fn cmd_compare_augmented(
    tx_spec: &TxSpec<'_>,
    new_inputs_str: &str,
    new_outputs_str: &str,
    block_size: usize,
    signed_method_str: &str,
) {
    let (label, tx) = resolve_tx(tx_spec);
    let signed_method = parse_signed_method(signed_method_str);
    let new_inputs = if new_inputs_str.is_empty() {
        Vec::new()
    } else {
        parse_values(new_inputs_str)
    };
    let new_outputs = if new_outputs_str.is_empty() {
        Vec::new()
    } else {
        parse_values(new_outputs_str)
    };

    if new_inputs.is_empty() && new_outputs.is_empty() {
        eprintln!("must provide at least one of --new-inputs or --new-outputs");
        std::process::exit(1);
    }

    let config = estimator::EstimatorConfig {
        lookup_k: block_size,
        ..Default::default()
    };
    let (before, after, delta) =
        estimator::compare_augmented(&tx, &new_inputs, &new_outputs, &config, signed_method);

    println!(
        "Base tx: {} ({} in / {} out)",
        label,
        tx.inputs.len(),
        tx.outputs.len()
    );
    if !new_inputs.is_empty() {
        println!("  + new inputs:  {:?}", new_inputs);
    }
    if !new_outputs.is_empty() {
        println!("  + new outputs: {:?}", new_outputs);
    }
    println!();
    println!("Mean signed log₂W per coin:");
    println!("  before: {:.4}", before);
    println!("  after:  {:.4}", after);
    println!("  delta:  {:+.4}", delta);
}

pub(crate) fn cmd_analyze_tx(inputs_str: &str, outputs_str: &str) {
    let tx = parse_tx(inputs_str, outputs_str);

    println!(
        "Transaction: {} inputs, {} outputs",
        tx.inputs.len(),
        tx.outputs.len()
    );
    println!("  Σinputs={}, Σoutputs={}", tx.input_sum(), tx.output_sum());
    println!("  Fee: {} sat", tx.fee());

    let metrics = mappings::analyze(&tx);
    println!("\nMappings:");
    println!("  Total: {}", metrics.total_mappings);
    println!("  Non-derived: {}", metrics.non_derived_count);
    println!("  Entropy: {:.3} bits", metrics.entropy);

    if !metrics.deterministic_links.is_empty() {
        println!(
            "  Deterministic links: {} pairs",
            metrics.deterministic_links.len()
        );
        for (i, o) in &metrics.deterministic_links {
            println!(
                "    input[{}]={} <-> output[{}]={}",
                i, tx.inputs[*i], o, tx.outputs[*o]
            );
        }
    }

    let all = mappings::enumerate_mappings(&tx);
    let non_derived = mappings::non_derived_mappings(&all);
    println!("\nNon-derived mappings:");
    for (idx, m) in non_derived.iter().enumerate() {
        println!("  Mapping {} ({} sub-txs):", idx, m.num_sub_txs());
        for (ins, outs) in m.input_sets.iter().zip(m.output_sets.iter()) {
            println!(
                "    {:?} -> {:?}  (sum={})",
                ins,
                outs,
                ins.iter().sum::<u64>()
            );
        }
    }
}

pub(crate) fn cmd_estimate(inputs_str: &str, outputs_str: &str, block_size: usize) {
    let tx = parse_tx(inputs_str, outputs_str);

    println!(
        "Transaction: {} inputs, {} outputs",
        tx.inputs.len(),
        tx.outputs.len()
    );

    let mut targets: Vec<u64> = tx.inputs.clone();
    targets.sort();
    targets.dedup();

    println!(
        "\n{:>12}  {:>12}  {:>12}  {:>12}",
        "Target E", "Sasamoto", "Lookup", "Brute(if≤25)"
    );
    println!("{:─<52}", "");

    let other_inputs = &tx.inputs;
    for &target in &targets {
        let sas = log_w_for_e_sat(other_inputs, target)
            .map(|lw| format!("{:.2}", lw))
            .unwrap_or_else(|| "N/A".into());

        let lookup = log_lookup_w(other_inputs, target, block_size)
            .map(|lw| format!("{:.2}", lw))
            .unwrap_or_else(|| "N/A".into());

        let brute = if other_inputs.len() <= 25 {
            format!("{}", brute_force_w(other_inputs, target))
        } else {
            "N>25".into()
        };

        println!("{:>12}  {:>12}  {:>12}  {:>12}", target, sas, lookup, brute);
    }
}

pub(crate) fn cmd_measure(inputs_str: &str, outputs_str: &str, block_size: usize) {
    let tx = parse_tx(inputs_str, outputs_str);

    let config = estimator::EstimatorConfig {
        lookup_k: block_size,
        ..Default::default()
    };

    let regime = estimator::analyze_regime(&tx, &config);
    println!(
        "Transaction: {} inputs, {} outputs",
        tx.inputs.len(),
        tx.outputs.len()
    );
    println!(
        "Regime: κ={:.3}, empirical={}, estimator={}",
        regime.kappa,
        fmt_regime(regime.empirical_regime),
        regime.estimator.as_str()
    );

    println!("\n{:>12}  {:>10}", "Input", "log₂W/N");
    println!("{:─<24}", "");
    for (i, &val) in tx.inputs.iter().enumerate() {
        let s = estimator::estimate_sub_tx(&tx, &[i], &config);
        println!("{:>12}  {:>10.4}", val, s);
    }

    let avg = estimator::estimate(&tx.inputs, tx.inputs.iter().sum::<u64>() / 2, &config);
    println!("\nMidpoint estimate: {:.4}", avg);
}

pub(crate) fn cmd_coin_measures(tx_spec: &TxSpec<'_>, block_size: usize, signed_method_str: &str) {
    let (label, tx) = resolve_tx(tx_spec);
    let signed_method = parse_signed_method(signed_method_str);

    let measurements = validation::per_coin_measurements(&tx, block_size, signed_method);
    validation::print_per_coin_measurements(&label, &tx, &measurements);
}

pub(crate) fn cmd_full_report(tx_spec: &TxSpec<'_>, block_size: usize, signed_method_str: &str) {
    let (label, tx) = resolve_tx(tx_spec);
    let signed_method = parse_signed_method(signed_method_str);
    let cfg = estimator::EstimatorConfig {
        lookup_k: block_size,
        ..Default::default()
    };
    let ln2 = 2f64.ln();

    let sum_in: u64 = tx.inputs.iter().sum();
    let sum_out: u64 = tx.outputs.iter().sum();
    let fee = sum_in.saturating_sub(sum_out);
    let n_in = tx.inputs.len();
    let n_out = tx.outputs.len();

    println!("═══ Full report: {} ═══", label);
    println!(
        "  N_in={}, N_out={}, Σin={}, Σout={}, fee={}",
        n_in, n_out, sum_in, sum_out, fee
    );

    let target_half = sum_in / 2;
    let regime = estimator::analyze_regime(&tx, &cfg);
    let dense_str = regime
        .dense_at_quartile
        .map(|b| b.to_string())
        .unwrap_or_else(|| "—".into());
    println!(
        "  κ = {:.4}, empirical = {}, dense_at_quartile = {}, estimator_picked = {}",
        regime.kappa,
        fmt_regime(regime.empirical_regime),
        dense_str,
        regime.estimator.as_str()
    );
    println!(
        "\n── W estimators at midpoint (target = Σin/2 = {}) ──",
        target_half
    );
    let lookup_mid = log_lookup_w(&tx.inputs, target_half, block_size);
    let sasamoto_mid = log_w_for_e_sat(&tx.inputs, target_half);
    println!(
        "  log W  [lookup, k={}]:   {}",
        block_size,
        fmt_log_w(lookup_mid, ln2)
    );
    println!(
        "  log W  [Sasamoto]:       {}",
        fmt_log_w(sasamoto_mid, ln2)
    );
    let mean_mid = estimator::estimate(&tx.inputs, target_half, &cfg);
    println!(
        "  estimator::estimate (log₂/N): {:.4}  (N={})",
        mean_mid, n_in
    );

    println!("\n── Per-coin ambiguity ──");
    let coins_sg = validation::per_coin_measurements(&tx, block_size, signed_method);
    per_coin_summary("Signed", &coins_sg, ln2);

    println!("\n── CJA mappings vs W (Maurer/Boltzmann) ──");
    for (name, fh) in [
        ("phantom", validation::FeeHandling::PhantomOutput),
        ("signed ", validation::FeeHandling::SignedMultiset),
    ] {
        match validation::compare_w_vs_mappings_with(&tx, &label, block_size, 26, fh, signed_method)
        {
            Some(mc) => {
                let sig = mc
                    .log_w_signed
                    .map(|v| format!("{:.2}", v / ln2))
                    .unwrap_or_else(|| "—".into());
                println!(
                    "  {}:  mappings={}, non-derived={}, max log₂W [sas={:.2}, lkp_in={:.2}, comb={:.2}, signed={}]",
                    name,
                    mc.n_mappings,
                    mc.n_non_derived,
                    mc.max_log_w_sasamoto / ln2,
                    mc.max_log_w_lookup / ln2,
                    mc.max_log_w_combined / ln2,
                    sig
                );
            }
            None => {
                println!("  {}:  skipped (too many coins for CJA at max=26)", name);
            }
        }
    }

    println!("\n── Denomination structure (inputs) ──");
    let distinguished: usize = tx.inputs.iter().filter(|&&v| is_distinguished(v)).count();
    let distinct_distinguished: std::collections::BTreeSet<u64> = tx
        .inputs
        .iter()
        .copied()
        .filter(|&v| is_distinguished(v))
        .collect();
    println!(
        "  empirical_regime = {}, distinguished = {}/{}, distinct_denoms = {}",
        fmt_regime(dense_subset_sum::empirical_regime(&tx.inputs)),
        distinguished,
        tx.inputs.len(),
        distinct_distinguished.len(),
    );
}

pub(crate) fn cmd_correlate_estimators(
    max_coins: usize,
    block_size: usize,
    signed_method_str: &str,
) {
    let signed_method = parse_signed_method(signed_method_str);
    let ln2 = 2f64.ln();
    let all: Vec<(&'static str, Transaction)> = fixtures::all_wasabi2_positive_cjtxs()
        .into_iter()
        .chain(fixtures::all_wasabi2_false_cjtxs())
        .collect();

    println!("tx_label\tN\tlog2_W_signed\tlog2_M\tper_coin_mean\tkappa");
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut pc = Vec::new();
    for (label, tx) in all {
        let n_total = tx.inputs.len() + tx.outputs.len();
        if n_total > max_coins {
            continue;
        }
        let mc = match validation::compare_w_vs_mappings_with(
            &tx,
            label,
            block_size,
            max_coins,
            validation::FeeHandling::SignedMultiset,
            signed_method,
        ) {
            Some(m) => m,
            None => continue,
        };
        let log_m = if mc.n_non_derived > 0 {
            (mc.n_non_derived as f64).ln() / ln2
        } else {
            continue;
        };
        let log_w_signed = match mc.log_w_signed {
            Some(v) if v.is_finite() => v / ln2,
            _ => continue,
        };
        let coins = validation::per_coin_measurements(&tx, block_size, signed_method);
        let vals: Vec<f64> = coins.iter().filter_map(|c| c.log_w_signed).collect();
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64 / ln2;
        let k = kappa(&tx.inputs).unwrap_or(f64::NAN);
        println!(
            "{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            label, n_total, log_w_signed, log_m, mean, k
        );
        xs.push(log_w_signed);
        ys.push(log_m);
        pc.push(mean);
    }

    let rho_ws = spearman_opt(&xs, &ys);
    let rho_pc = spearman_opt(&pc, &ys);
    println!();
    println!(
        "n = {} txs with N_in+N_out ≤ {} and finite ground truth",
        xs.len(),
        max_coins
    );
    match rho_ws {
        Some(r) => println!("spearman(log2_W_signed, log2_M) = {:.4}", r),
        None => println!("spearman(log2_W_signed, log2_M) = NA (insufficient data)"),
    }
    match rho_pc {
        Some(r) => println!("spearman(per_coin_mean,  log2_M) = {:.4}", r),
        None => println!("spearman(per_coin_mean,  log2_M) = NA (insufficient data)"),
    }
}

pub(crate) fn cmd_suggest_split(
    inputs_str: &str,
    outputs_str: &str,
    change: u64,
    max_pieces: usize,
    block_size: usize,
    signed_method_str: &str,
) {
    if change == 0 {
        eprintln!("error: change must be > 0");
        std::process::exit(1);
    }
    let signed_method = parse_signed_method(signed_method_str);
    let inputs = parse_values(inputs_str);
    let outputs = parse_values(outputs_str);
    let tx = Transaction::new(inputs.clone(), outputs.clone());

    let (best_split, best_mean) =
        change_split::suggest_output_split(&tx, change, max_pieces, block_size, signed_method);

    println!(
        "Tx: inputs={:?}, outputs (excl. change)={:?}",
        inputs, outputs
    );
    println!(
        "Change to decompose: {} sat, max_pieces={}",
        change, max_pieces
    );
    println!();
    println!("Best split: {:?}", best_split);
    println!(
        "  → {} piece(s), sum={}, Hamming weight profile: {:?}",
        best_split.len(),
        best_split.iter().sum::<u64>(),
        best_split
            .iter()
            .map(|v| v.count_ones())
            .collect::<Vec<_>>(),
    );
    if best_mean.is_finite() {
        println!("  Mean signed log₂ W across all coins: {:.4}", best_mean);
    } else {
        println!("  (single-output case — mean undefined)");
    }
}
