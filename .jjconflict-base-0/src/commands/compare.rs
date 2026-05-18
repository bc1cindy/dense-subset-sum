//! W(E)-vs-ground-truth + W-vs-mappings benchmark subcommands.

use dense_subset_sum::{comparison, fixtures, validation};

use super::{parse_signed_method, parse_values};

pub(crate) fn cmd_compare(values_str: &str, min_w: u64, block_size: usize, csv: bool) {
    let a = parse_values(values_str);
    if a.len() > 25 {
        eprintln!("Warning: N={} > 25, brute force will be very slow", a.len());
    }
    let report = comparison::compare(&a, min_w, block_size, 10_000_000, "user_input");
    if csv {
        comparison::print_report_csv(&report);
    } else {
        comparison::print_report(&report);
    }
}

pub(crate) fn cmd_compare_fixtures(min_w: u64, block_size: usize) {
    for (label, values) in fixtures::all_comparison_sets() {
        let report = comparison::compare(&values, min_w, block_size, 10_000_000, label);
        comparison::print_report(&report);
    }
}

pub(crate) fn cmd_compare_random(
    n: usize,
    max_val: u64,
    seed: u64,
    min_w: u64,
    block_size: usize,
    matrix: bool,
) {
    if matrix {
        let configs: Vec<(usize, u64)> = vec![
            (8, 100),
            (8, 1_000),
            (8, 100_000),
            (12, 100),
            (12, 1_000),
            (12, 100_000),
            (16, 100),
            (16, 1_000),
            (16, 100_000),
            (20, 100),
            (20, 1_000),
            (20, 100_000),
        ];
        let seeds: Vec<u64> = (0..5).collect();
        let mut batch_rows = Vec::new();

        for &(n, max_val) in &configs {
            let mut reports = Vec::new();
            for &s in &seeds {
                let values = comparison::uniform_random_set(n, max_val, s);
                let label = format!("uniform_N{}_range{}_seed{}", n, max_val, s);
                let report = comparison::compare(&values, min_w, block_size, 10_000_000, &label);
                reports.push(report);
            }
            let config_label = format!("N={} range={}", n, max_val);
            batch_rows.push(comparison::aggregate_reports(&reports, &config_label));
        }
        comparison::print_batch_summary(&batch_rows);
    } else {
        if n > 25 {
            eprintln!("Warning: N={} > 25, brute force will be very slow", n);
        }
        let values = comparison::uniform_random_set(n, max_val, seed);
        let label = format!("uniform_N{}_range{}_seed{}", n, max_val, seed);
        let report = comparison::compare(&values, min_w, block_size, 10_000_000, &label);
        comparison::print_report(&report);
    }
}

pub(crate) fn cmd_compare_wasabi2(
    max_coins: usize,
    block_size: usize,
    fee_handling_str: &str,
    signed_method_str: &str,
) {
    let fee_handling = match fee_handling_str {
        "phantom" => validation::FeeHandling::PhantomOutput,
        "signed" => validation::FeeHandling::SignedMultiset,
        other => {
            eprintln!(
                "invalid --fee-handling value: {:?} (expected \"phantom\" or \"signed\")",
                other
            );
            std::process::exit(1);
        }
    };
    let signed_method = parse_signed_method(signed_method_str);

    let txs = fixtures::all_wasabi2_false_cjtxs();
    let mut comparisons = Vec::new();
    let mut skipped = 0;

    println!(
        "fee-handling: {}  signed-method: {}",
        fee_handling_str, signed_method_str
    );
    for (label, tx) in &txs {
        match validation::compare_w_vs_mappings_with(
            tx,
            label,
            block_size,
            max_coins,
            fee_handling,
            signed_method,
        ) {
            Some(mc) => {
                validation::print_mapping_comparison(&mc);
                comparisons.push(mc);
            }
            None => {
                // Coin count includes phantom fee output (both variants enumerate against it).
                let total = tx.inputs.len() + tx.outputs.len() + 1;
                println!("SKIP {} ({} coins > max {})", label, total, max_coins);
                skipped += 1;
            }
        }
    }

    if !comparisons.is_empty() {
        validation::print_mapping_summary(&comparisons);
        if let Some(corr) = validation::correlate_w_vs_mappings(&comparisons) {
            validation::print_mapping_correlation(&corr);
        }
    }
    println!(
        "\n{} txs analyzed, {} skipped (>{} coins)",
        comparisons.len(),
        skipped,
        max_coins
    );
}

pub(crate) fn cmd_compare_synthetic(
    n: usize,
    l_max: u64,
    seed: u64,
    min_w: u64,
    block_size: usize,
    dp_max: usize,
) {
    let values = comparison::uniform_random_set(n, l_max, seed);
    let sum: u64 = values.iter().sum();
    println!(
        "Synthetic uniform inputs: N={}, L∈[1,{}], seed={}, Σa={}",
        n, l_max, seed, sum
    );
    if sum as usize > dp_max {
        eprintln!(
            "error: Σa = {} exceeds dp_max = {}. Lower --n or --l-max, or raise --dp-max.",
            sum, dp_max
        );
        std::process::exit(1);
    }
    let label = format!("synthetic_uniform_N{}_L{}_seed{}", n, l_max, seed);
    let report =
        match comparison::compare_dp_ground_truth(&values, min_w, block_size, dp_max, &label) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        };
    comparison::print_report(&report);
}

pub(crate) fn cmd_validate(values_str: &str, min_w: u64, block_size: usize) {
    let a = parse_values(values_str);
    if a.len() > 25 {
        eprintln!("Warning: N={} > 25, brute force will be slow", a.len());
    }

    let report = comparison::compare(&a, min_w, block_size, 10_000_000, "validate");

    for summary in [&report.sasamoto, &report.lookup, &report.dp] {
        println!("{}:", summary.name);
        println!("  Points tested: {}", summary.n_points);
        println!("  Median error: {:.1}%", summary.median_error * 100.0);
        println!("  Spearman ρ: {:.4}", summary.spearman);
    }

    if report.rows.len() <= 10 {
        for row in &report.rows {
            println!(
                "    E={}: exact={:.0}, sas={:?}, lkp={:?}, dp={:?}",
                row.e_target, row.w_exact, row.w_sasamoto, row.w_lookup, row.w_dp
            );
        }
    }
}
