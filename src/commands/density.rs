//! κ/κ_c regime tooling, target sweeps, and subset-size density sweeps.

use dense_subset_sum::{
    LookupConfig, density_regime, empirical_regime, estimator, find_dense_region, fixtures,
    kappa, kappa_c, log_w_for_e, n_c, sumset_size_with_config, validation,
};

use super::{TxSpec, parse_values, parse_values_f64, resolve_tx, resolve_values};

pub(crate) fn cmd_density(
    tx_label: Option<&str>,
    tx_json: Option<&std::path::Path>,
    values_str: &str,
    target: Option<u64>,
    block_size: usize,
) {
    let (label, values) = resolve_values(tx_label, tx_json, values_str, false);

    let cfg = estimator::EstimatorConfig {
        lookup_k: block_size,
        ..Default::default()
    };
    let ln2 = 2f64.ln();

    let sum: u64 = values.iter().sum();
    let e_target = target.unwrap_or(sum / 2);

    let de = estimator::estimate_density(&values, e_target, &cfg);

    println!("═══ Density: {} ═══", label);
    println!(
        "  N = {}, Σ = {}, E_target = {}{}",
        values.len(),
        sum,
        e_target,
        if target.is_none() {
            " (default: Σ/2)"
        } else {
            ""
        }
    );
    match de.kappa {
        Some(k) => println!("  κ = {:.4}", k),
        None => println!("  κ = —"),
    }
    println!("  Regime:          {:?}", de.regime);
    println!("  Estimator used:  {}", de.estimator_used.as_str());
    match de.log_w {
        Some(lw) if lw.is_finite() => {
            println!(
                "  log W  (nat):   {:.4}   (log₂ W = {:.4},  W ≈ 2^{:.2})",
                lw,
                lw / ln2,
                lw / ln2
            );
        }
        Some(_) => println!("  log W:           −∞"),
        None => println!("  log W:           —"),
    }
    println!("  Reliable:        {}", de.reliable);
    println!(
        "  is_dense(...):   {}",
        de.regime == estimator::Regime::Dense
    );
}

pub(crate) fn cmd_density_scan(values_str: &str, steps: usize, min_log_w: f64) {
    let a = parse_values_f64(values_str);
    let e_max: f64 = a.iter().sum();

    println!("N={}, E_max={}", a.len(), e_max);
    println!("{:>10}  {:>10}  {:>14}", "E", "log W(E)", "W(E)");
    println!("{:─<38}", "");

    for i in 1..steps {
        let e = e_max * i as f64 / steps as f64;
        if let Some(lw) = log_w_for_e(&a, e) {
            let w_str = if lw < 700.0 {
                format!("{:.2}", lw.exp())
            } else {
                "∞".into()
            };
            println!("{:>10.1}  {:>10.4}  {:>14}", e, lw, w_str);
        }
    }

    if let Some(r) = find_dense_region(&a, min_log_w, 1000) {
        println!("\nDense region (log W >= {:.1}):", min_log_w);
        println!("  Peak: E={:.1}, W~={:.1}", r.e_peak, r.w_peak);
        println!("  Range: [{:.1}, {:.1}]", r.e_lo, r.e_hi);
        println!("  Safe E: {:.1}, log W={:.2}", r.e_safe, r.log_w_safe);
    }
}

pub(crate) fn cmd_kappa(values_str: &str, target: Option<u64>) {
    let a = parse_values(values_str);

    let k = kappa(&a).unwrap_or(f64::NAN);
    println!("N = {}", a.len());
    println!("max(A) = {}", a.iter().max().unwrap_or(&0));
    println!("κ = log₂(max(A)) / N = {:.4}", k);

    if let Some(e) = target {
        let l = *a.iter().max().unwrap() as f64;
        let n = a.len() as f64;
        let x = e as f64 / (n * l);
        println!("\nTarget E = {}", e);
        println!("x = E/(N·L) = {:.4}", x);

        if let Some(kc) = kappa_c(x) {
            println!("κ_c(x) = {:.4}", kc);
            let dense = k < kc;
            println!(
                "κ {} κ_c → {}",
                if dense { "<" } else { "≥" },
                if dense { "DENSE" } else { "SPARSE" }
            );
        }

        if let Some((k_val, kc_val, is_dense)) = density_regime(&a, e) {
            println!(
                "\ndensity_regime: κ={:.4}, κ_c={:.4}, dense={}",
                k_val, kc_val, is_dense
            );
        }
    }
}

pub(crate) fn cmd_dense_boundary(tx_spec: &TxSpec<'_>, samples: usize, seed: u64) {
    let (label, tx) = resolve_tx(tx_spec);
    let a = &tx.inputs;
    let n = a.len();
    let sum: u64 = a.iter().sum();
    let k = kappa(a).unwrap_or(f64::NAN);

    println!("tx: {}", label);
    println!("N_in = {}, Σa = {}, κ = {:.4}", n, sum, k);

    let steps = 512usize;
    let mut dense_count = 0usize;
    let mut e_lo: Option<u64> = None;
    let mut e_hi: Option<u64> = None;
    for i in 1..steps {
        let e = ((sum as u128) * (i as u128) / (steps as u128)) as u64;
        if e == 0 || e >= sum {
            continue;
        }
        if let Some((_, _, true)) = density_regime(a, e) {
            dense_count += 1;
            e_lo.get_or_insert(e);
            e_hi = Some(e);
        }
    }
    let frac = dense_count as f64 / (steps - 2) as f64;
    match (e_lo, e_hi) {
        (Some(lo), Some(hi)) => println!(
            "target-space dense range: E ∈ [{}, {}]  (fraction of [0,Σa] = {:.4})",
            lo, hi, frac
        ),
        _ => println!(
            "target-space dense range: ∅  (no E in [0,Σa] is dense at current scan resolution)"
        ),
    }

    if n < 2 {
        println!("subset-size crossover: N too small");
        return;
    }
    let mut sizes: Vec<usize> = Vec::new();
    let mut s = 2usize;
    while s < n {
        sizes.push(s);
        s *= 2;
    }
    sizes.push(n);
    let rows = validation::subset_density_sweep(a, &sizes, samples, seed);
    println!("subset-size sweep:");
    println!("  k\tfraction_dense");
    let mut prev: Option<(usize, f64)> = None;
    let mut cross: Option<f64> = None;
    for r in &rows {
        println!("  {}\t{:.4}", r.size, r.fraction_dense);
        if let Some((pk, pf)) = prev
            && (pf - 0.5).signum() != (r.fraction_dense - 0.5).signum()
            && pf.is_finite()
            && r.fraction_dense.is_finite()
        {
            let t = (0.5 - pf) / (r.fraction_dense - pf);
            cross = Some(pk as f64 + t * (r.size as f64 - pk as f64));
        }
        prev = Some((r.size, r.fraction_dense));
    }
    match cross {
        Some(kc) => println!("crossover k* (fraction_dense ≈ 0.5): {:.2}", kc),
        None => {
            println!("crossover k*: not found in sampled range (fraction_dense never crosses 0.5)")
        }
    }
}

pub(crate) fn cmd_subset_density(
    tx_label: Option<&str>,
    tx_json: Option<&std::path::Path>,
    values_str: &str,
    all_coins: bool,
    sizes_str: Option<&str>,
    samples: usize,
    seed: u64,
) {
    let (label, values) = resolve_values(tx_label, tx_json, values_str, all_coins);

    let sizes: Vec<usize> = match sizes_str {
        Some(s) => s
            .split(',')
            .map(|v| v.trim().parse::<usize>().expect("invalid size"))
            .collect(),
        None => {
            let n = values.len();
            let mut s = Vec::new();
            let mut k = 4usize;
            while k <= n {
                s.push(k);
                k *= 2;
            }
            if !s.contains(&n) && n >= 4 {
                s.push(n);
            }
            s
        }
    };

    let rows = validation::subset_density_sweep(&values, &sizes, samples, seed);
    validation::print_subset_density_sweep(&label, values.len(), &rows);
}

pub(crate) fn cmd_empirical_nc(block_size: usize, max_entries: usize) {
    let cfg = LookupConfig::from_max_entries(max_entries);
    println!("# label              fixture id (w2_… negatives, w2pos_… positives)");
    println!("# class              neg = false coinjoin, pos = real Wasabi2 coinjoin");
    println!("# N                  number of inputs");
    println!("# nc                 Sasamoto critical size ½·log₂(π/2·Σaⱼ²) (paper A.7)");
    println!("# nc_over_n          nc/N; saddle is reliable when ≪ 1 (gate threshold τ=0.5)");
    println!("# sumset_size        distinct reachable sums up to block_size convolution");
    println!("# sumset_saturated   true when sumset_size hit the max_entries cap");
    println!("# regime             EqualAmount / RadixGeometric / Arithmetic / PathologicalBatch");
    println!("label\tclass\tN\tnc\tnc_over_n\tsumset_size\tsumset_saturated\tregime");
    let all = fixtures::all_wasabi2_false_cjtxs()
        .into_iter()
        .map(|(l, tx)| ("neg", l, tx))
        .chain(
            fixtures::all_wasabi2_positive_cjtxs()
                .into_iter()
                .map(|(l, tx)| ("pos", l, tx)),
        );
    for (class, label, tx) in all {
        let a = &tx.inputs;
        let n = a.len();
        let nc = n_c(a);
        let sumset = sumset_size_with_config(a, block_size, &cfg);
        let saturated = sumset >= cfg.max_entries;
        let regime = empirical_regime(a)
            .map(|r| format!("{:?}", r))
            .unwrap_or_else(|| "—".into());
        println!(
            "{}\t{}\t{}\t{:.3}\t{:.4}\t{}\t{}\t{}",
            label,
            class,
            n,
            nc,
            nc / n as f64,
            sumset,
            saturated,
            regime,
        );
    }
}
