//! Clap definitions and dispatcher. `main.rs` only parses and calls [`run`].

use clap::{Parser, Subcommand};

use crate::commands::{
    TxSpec, cmd_analyze_tx, cmd_coin_measures, cmd_compare, cmd_compare_augmented,
    cmd_compare_fixtures, cmd_compare_random, cmd_compare_synthetic, cmd_compare_wasabi2,
    cmd_correlate_estimators, cmd_dense_boundary, cmd_density, cmd_density_scan, cmd_estimate,
    cmd_full_report, cmd_kappa, cmd_measure, cmd_subset_density, cmd_suggest_split, cmd_validate,
};

#[derive(Parser)]
#[command(
    name = "dense-subset-sum",
    about = "Subset sum density analysis for CoinJoin privacy",
    after_help = "See README.md for quick-start commands and interpretation guide."
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan W(E) across the energy range for a set of values
    DensityScan {
        /// Comma-separated values
        #[arg(short, long)]
        values: String,
        /// Number of scan points
        #[arg(short, long, default_value = "20")]
        steps: usize,
        /// Minimum log W threshold for dense region
        #[arg(short, long, default_value = "2.3")]
        min_log_w: f64,
    },
    /// Enumerate mappings and compute Boltzmann metrics for a transaction
    AnalyzeTx {
        /// Comma-separated input values (satoshis)
        #[arg(short, long)]
        inputs: String,
        /// Comma-separated output values (satoshis)
        #[arg(short, long)]
        outputs: String,
    },
    /// Estimate W(E) using Sasamoto + lookup table
    Estimate {
        /// Comma-separated input values (satoshis)
        #[arg(short, long)]
        inputs: String,
        /// Comma-separated output values (satoshis)
        #[arg(short, long)]
        outputs: String,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "15")]
        block_size: usize,
    },
    /// Cross-validate estimators against brute force
    Validate {
        /// Comma-separated values
        #[arg(short, long)]
        values: String,
        /// Minimum W for test points
        #[arg(short, long, default_value = "10")]
        min_w: u64,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
    },
    /// Compute κ and κ_c for a set of values
    Kappa {
        /// Comma-separated values
        #[arg(short, long)]
        values: String,
        /// Target energy (optional, for κ_c computation)
        #[arg(short, long)]
        target: Option<u64>,
    },
    /// Per-coin mean signed log₂W for a transaction
    Measure {
        /// Comma-separated input values (satoshis)
        #[arg(short, long)]
        inputs: String,
        /// Comma-separated output values (satoshis)
        #[arg(short, long)]
        outputs: String,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "15")]
        block_size: usize,
    },
    /// Compare W estimators against brute force for given values
    Compare {
        /// Comma-separated values
        #[arg(short, long)]
        values: String,
        /// Minimum W for test points
        #[arg(short, long, default_value = "5")]
        min_w: u64,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// Output as CSV
        #[arg(long)]
        csv: bool,
    },
    /// Run comparison on all built-in fixtures
    CompareFixtures {
        /// Minimum W for test points
        #[arg(short, long, default_value = "5")]
        min_w: u64,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
    },
    /// Compare estimators on random uniform values
    CompareRandom {
        /// Number of values
        #[arg(short, long, default_value = "16")]
        n: usize,
        /// Maximum value (uniform range [1, max_val])
        #[arg(short, long, default_value = "1000")]
        max_val: u64,
        /// Random seed
        #[arg(short, long, default_value = "42")]
        seed: u64,
        /// Minimum W for test points
        #[arg(short = 'w', long, default_value = "10")]
        min_w: u64,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// Run full matrix (overrides n/max_val/seed)
        #[arg(long)]
        matrix: bool,
    },
    /// Compare W estimates vs CJA mappings for real Wasabi 2 transactions
    CompareWasabi2 {
        /// Max total coins for CJA enumeration (default 26)
        #[arg(short, long, default_value = "26")]
        max_coins: usize,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// "phantom" (Maurer canonical fee-as-output) or "signed" (signed W on fee).
        #[arg(long, default_value = "phantom")]
        fee_handling: String,
        /// Signed probe estimator: "sasamoto" or "lookup". Required when
        /// --fee-handling=signed. No default policy — caller picks based on deployment.
        #[arg(long)]
        signed_method: String,
    },
    /// Per-coin W estimates (signed ±multiset probe) for every input and output.
    CoinMeasures {
        /// Wasabi2 fixture label (e.g. w2_6a6dcc22_17in6out). Overrides --inputs/--outputs.
        #[arg(long)]
        tx_label: Option<String>,
        /// Path to JSON file: {"label":..., "inputs":[...], "outputs":[...]}. Overrides --tx-label.
        #[arg(long)]
        tx_json: Option<std::path::PathBuf>,
        /// Comma-separated input values (satoshis). Ignored if --tx-label given.
        #[arg(short, long, default_value = "")]
        inputs: String,
        /// Comma-separated output values (satoshis). Ignored if --tx-label given.
        #[arg(short, long, default_value = "")]
        outputs: String,
        /// Block size for lookup table
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// Signed probe estimator: "sasamoto" or "lookup". Required — no library-level
        /// default policy; caller picks based on deployment (WASM/mobile/server).
        #[arg(long)]
        signed_method: String,
    },
    /// Random-subset density sweep: fraction of subsets with κ < κ_c per size.
    SubsetDensity {
        /// Wasabi2 fixture label (e.g. w2_37e11e3f_159in3out). Overrides --values.
        #[arg(long)]
        tx_label: Option<String>,
        /// Path to JSON tx file. Uses tx.inputs (or inputs+outputs if --all-coins).
        #[arg(long)]
        tx_json: Option<std::path::PathBuf>,
        /// Comma-separated values. Ignored if --tx-label given.
        #[arg(short, long, default_value = "")]
        values: String,
        /// Use all coins (inputs+outputs) of the fixture, not just inputs.
        #[arg(long, default_value = "false")]
        all_coins: bool,
        /// Comma-separated subset sizes (defaults to 4,8,16,32,64,... up to N).
        #[arg(long)]
        sizes: Option<String>,
        /// Number of random subsets per size.
        #[arg(long, default_value = "200")]
        samples: usize,
        /// Random seed.
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// Spearman correlation: cheap estimators vs Boltzmann ground truth (log |M_non_derived|).
    CorrelateEstimators {
        #[arg(long, default_value = "26")]
        max_coins: usize,
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// Signed probe estimator: "sasamoto" or "lookup".
        #[arg(long)]
        signed_method: String,
    },
    /// Dense target range [E_lo, E_hi], fraction of [0,Σa] dense, and k* crossover.
    DenseBoundary {
        #[arg(long)]
        tx_label: Option<String>,
        /// Path to JSON tx file. Overrides --tx-label.
        #[arg(long)]
        tx_json: Option<std::path::PathBuf>,
        #[arg(short, long, default_value = "")]
        inputs: String,
        #[arg(short, long, default_value = "")]
        outputs: String,
        #[arg(long, default_value = "200")]
        samples: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// Kitchen-sink report: every estimator side-by-side on a single transaction.
    FullReport {
        /// Wasabi2 fixture label (positive or negative). Overrides --inputs/--outputs.
        #[arg(long)]
        tx_label: Option<String>,
        /// Path to JSON file: {"label":..., "inputs":[...], "outputs":[...]}. Overrides --tx-label.
        #[arg(long)]
        tx_json: Option<std::path::PathBuf>,
        /// Comma-separated input values (satoshis). Ignored if --tx-label given.
        #[arg(short, long, default_value = "")]
        inputs: String,
        /// Comma-separated output values (satoshis). Ignored if --tx-label given.
        #[arg(short, long, default_value = "")]
        outputs: String,
        /// Block size for lookup table.
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// Signed probe estimator: "sasamoto" or "lookup". Only applied
        /// to the SignedMultiset fee-handling row.
        #[arg(long)]
        signed_method: String,
    },
    /// Unified density API: 3-tier scale switch (exact DP → lookup → Sasamoto).
    Density {
        /// Wasabi2 fixture label (uses tx.inputs). Overrides --values.
        #[arg(long)]
        tx_label: Option<String>,
        /// Path to JSON tx file. Uses tx.inputs. Overrides --tx-label.
        #[arg(long)]
        tx_json: Option<std::path::PathBuf>,
        /// Comma-separated values (satoshis). Ignored if --tx-label given.
        #[arg(short, long, default_value = "")]
        values: String,
        /// Target energy E (subset sum). Defaults to Σvalues / 2.
        #[arg(short, long)]
        target: Option<u64>,
        /// Block size for lookup table.
        #[arg(short = 'k', long, default_value = "15")]
        block_size: usize,
    },
    /// Sasamoto vs DP ground truth on synthetic uniform inputs at large N.
    CompareSynthetic {
        /// Number of values. N≥100 slows because Lookup scales as N·Σa.
        #[arg(short, long, default_value = "50")]
        n: usize,
        /// Values drawn uniformly from [1, l_max].
        #[arg(long, default_value = "500")]
        l_max: u64,
        /// Random seed.
        #[arg(short, long, default_value = "42")]
        seed: u64,
        /// Minimum W for included test points.
        #[arg(short = 'w', long, default_value = "100")]
        min_w: u64,
        /// Block size for lookup table.
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// Max DP table size (Σa must fit). Default 10M cells ≈ 160 MB u128.
        #[arg(long, default_value = "10000000")]
        dp_max: usize,
    },
    /// Mean signed log₂W per coin: augmented tx vs base tx (reports before/after/delta).
    CompareAugmented {
        /// Wasabi2 fixture label for the current tx. Overrides --inputs/--outputs.
        #[arg(long)]
        tx_label: Option<String>,
        /// JSON file: {"label":..., "inputs":[...], "outputs":[...]} for the current tx.
        #[arg(long)]
        tx_json: Option<std::path::PathBuf>,
        /// Current tx inputs (satoshis). Ignored if --tx-label/--tx-json given.
        #[arg(short, long, default_value = "")]
        inputs: String,
        /// Current tx outputs (satoshis). Ignored if --tx-label/--tx-json given.
        #[arg(short, long, default_value = "")]
        outputs: String,
        /// Inputs you are considering adding (satoshis, comma-separated).
        #[arg(long, default_value = "")]
        new_inputs: String,
        /// Outputs you are considering adding (satoshis, comma-separated).
        #[arg(long, default_value = "")]
        new_outputs: String,
        /// Block size for lookup table in the signed probe.
        #[arg(short = 'k', long, default_value = "10")]
        block_size: usize,
        /// Signed probe estimator: "sasamoto" or "lookup". Required.
        #[arg(long)]
        signed_method: String,
    },
    /// Rank low-Hamming-weight change decompositions by mean signed log₂W.
    SuggestSplit {
        /// Comma-separated input values (satoshis).
        #[arg(short, long)]
        inputs: String,
        /// Comma-separated output values, EXCLUDING the change being split.
        #[arg(short, long)]
        outputs: String,
        /// Change amount to decompose (satoshis).
        #[arg(short, long)]
        change: u64,
        /// Maximum number of pieces in the split (1..=8).
        #[arg(short = 'p', long, default_value = "4")]
        max_pieces: usize,
        /// Block size for lookup table in the signed probe.
        #[arg(short = 'k', long, default_value = "3")]
        block_size: usize,
        /// Signed probe estimator: "sasamoto" or "lookup". Required.
        #[arg(long)]
        signed_method: String,
    },
}

pub fn run(cli: Cli) {
    match cli.command {
        Command::DensityScan {
            values,
            steps,
            min_log_w,
        } => cmd_density_scan(&values, steps, min_log_w),
        Command::AnalyzeTx { inputs, outputs } => cmd_analyze_tx(&inputs, &outputs),
        Command::Estimate {
            inputs,
            outputs,
            block_size,
        } => cmd_estimate(&inputs, &outputs, block_size),
        Command::Validate {
            values,
            min_w,
            block_size,
        } => cmd_validate(&values, min_w, block_size),
        Command::Kappa { values, target } => cmd_kappa(&values, target),
        Command::Measure {
            inputs,
            outputs,
            block_size,
        } => cmd_measure(&inputs, &outputs, block_size),
        Command::Compare {
            values,
            min_w,
            block_size,
            csv,
        } => cmd_compare(&values, min_w, block_size, csv),
        Command::CompareFixtures { min_w, block_size } => cmd_compare_fixtures(min_w, block_size),
        Command::CompareRandom {
            n,
            max_val,
            seed,
            min_w,
            block_size,
            matrix,
        } => cmd_compare_random(n, max_val, seed, min_w, block_size, matrix),
        Command::CompareWasabi2 {
            max_coins,
            block_size,
            fee_handling,
            signed_method,
        } => cmd_compare_wasabi2(max_coins, block_size, &fee_handling, &signed_method),
        Command::CoinMeasures {
            tx_label,
            tx_json,
            inputs,
            outputs,
            block_size,
            signed_method,
        } => cmd_coin_measures(
            &TxSpec {
                label: tx_label.as_deref(),
                json: tx_json.as_deref(),
                inputs_str: &inputs,
                outputs_str: &outputs,
            },
            block_size,
            &signed_method,
        ),
        Command::SubsetDensity {
            tx_label,
            tx_json,
            values,
            all_coins,
            sizes,
            samples,
            seed,
        } => cmd_subset_density(
            tx_label.as_deref(),
            tx_json.as_deref(),
            &values,
            all_coins,
            sizes.as_deref(),
            samples,
            seed,
        ),
        Command::CompareSynthetic {
            n,
            l_max,
            seed,
            min_w,
            block_size,
            dp_max,
        } => cmd_compare_synthetic(n, l_max, seed, min_w, block_size, dp_max),
        Command::CompareAugmented {
            tx_label,
            tx_json,
            inputs,
            outputs,
            new_inputs,
            new_outputs,
            block_size,
            signed_method,
        } => cmd_compare_augmented(
            &TxSpec {
                label: tx_label.as_deref(),
                json: tx_json.as_deref(),
                inputs_str: &inputs,
                outputs_str: &outputs,
            },
            &new_inputs,
            &new_outputs,
            block_size,
            &signed_method,
        ),
        Command::SuggestSplit {
            inputs,
            outputs,
            change,
            max_pieces,
            block_size,
            signed_method,
        } => cmd_suggest_split(
            &inputs,
            &outputs,
            change,
            max_pieces,
            block_size,
            &signed_method,
        ),
        Command::CorrelateEstimators {
            max_coins,
            block_size,
            signed_method,
        } => cmd_correlate_estimators(max_coins, block_size, &signed_method),
        Command::DenseBoundary {
            tx_label,
            tx_json,
            inputs,
            outputs,
            samples,
            seed,
        } => cmd_dense_boundary(
            &TxSpec {
                label: tx_label.as_deref(),
                json: tx_json.as_deref(),
                inputs_str: &inputs,
                outputs_str: &outputs,
            },
            samples,
            seed,
        ),
        Command::FullReport {
            tx_label,
            tx_json,
            inputs,
            outputs,
            block_size,
            signed_method,
        } => cmd_full_report(
            &TxSpec {
                label: tx_label.as_deref(),
                json: tx_json.as_deref(),
                inputs_str: &inputs,
                outputs_str: &outputs,
            },
            block_size,
            &signed_method,
        ),
        Command::Density {
            tx_label,
            tx_json,
            values,
            target,
            block_size,
        } => cmd_density(
            tx_label.as_deref(),
            tx_json.as_deref(),
            &values,
            target,
            block_size,
        ),
    }
}
