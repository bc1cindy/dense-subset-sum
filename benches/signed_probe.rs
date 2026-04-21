//! Wall-time + peak-allocation harness for the signed probe.
//!
//! Time is measured with an adaptive loop: the function is called repeatedly
//! until ~5 ms of wall-time elapses, then the total is divided by the iteration
//! count. Fast cells (µs) accumulate thousands of iterations for precision;
//! slow cells (hundreds of ms) do a single call. Memory is measured in the
//! same pass via `peak_alloc::PeakAlloc` as the global allocator.
//!
//! Output is TSV on stdout with columns `regime`, `target`, `N`, `method`,
//! `median_ns`, `peak_bytes`, `result_is_some`. The default grid is trimmed
//! to the cells that carry signal for the pick-per-deployment decision
//! (`dense` + `stdenom` × `mid` + `q1` × {16,32,64,128} × k=8 = 32 cells).
//! Widen via env vars (comma-separated):
//!
//!     BENCH_REGIMES=sparse,dense,equal,stdenom
//!     BENCH_TARGETS=mid,q1
//!     BENCH_NS=8,16,32,64,128,256
//!     BENCH_BLOCK_SIZES=8,10,12,14

use dense_subset_sum::{
    LookupConfig, log_lookup_w_signed_target_aware_with_config, log_w_signed_sasamoto,
};
use peak_alloc::PeakAlloc;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant};

#[global_allocator]
static PEAK: PeakAlloc = PeakAlloc;

const SEED: u64 = 42;
const DEFAULT_NS: &[usize] = &[16, 32, 64, 128];
const DEFAULT_BLOCK_SIZES: &[usize] = &[8];
const DEFAULT_REGIMES: &[Regime] = &[Regime::Dense, Regime::Stdenom];
const DEFAULT_TARGETS: &[TargetPos] = &[TargetPos::Midpoint, TargetPos::Quartile];

fn load_usizes(var: &str, fallback: &[usize]) -> Vec<usize> {
    std::env::var(var)
        .ok()
        .and_then(|s| {
            let v: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if v.is_empty() { None } else { Some(v) }
        })
        .unwrap_or_else(|| fallback.to_vec())
}

fn load_regimes() -> Vec<Regime> {
    std::env::var("BENCH_REGIMES")
        .ok()
        .and_then(|s| {
            let v: Vec<Regime> = s
                .split(',')
                .filter_map(|x| Regime::from_label(x.trim()))
                .collect();
            if v.is_empty() { None } else { Some(v) }
        })
        .unwrap_or_else(|| DEFAULT_REGIMES.to_vec())
}

fn load_targets() -> Vec<TargetPos> {
    std::env::var("BENCH_TARGETS")
        .ok()
        .and_then(|s| {
            let v: Vec<TargetPos> = s
                .split(',')
                .filter_map(|x| TargetPos::from_label(x.trim()))
                .collect();
            if v.is_empty() { None } else { Some(v) }
        })
        .unwrap_or_else(|| DEFAULT_TARGETS.to_vec())
}

#[derive(Copy, Clone)]
enum Regime {
    Sparse,
    Dense,
    EqualDenom,
    Stdenom,
}

impl Regime {
    fn label(self) -> &'static str {
        match self {
            Regime::Sparse => "sparse",
            Regime::Dense => "dense",
            Regime::EqualDenom => "equal",
            Regime::Stdenom => "stdenom",
        }
    }

    fn from_label(s: &str) -> Option<Self> {
        match s {
            "sparse" => Some(Regime::Sparse),
            "dense" => Some(Regime::Dense),
            "equal" => Some(Regime::EqualDenom),
            "stdenom" => Some(Regime::Stdenom),
            _ => None,
        }
    }
}

#[derive(Copy, Clone)]
enum TargetPos {
    Midpoint,
    Quartile,
}

impl TargetPos {
    fn label(self) -> &'static str {
        match self {
            TargetPos::Midpoint => "mid",
            TargetPos::Quartile => "q1",
        }
    }

    fn from_label(s: &str) -> Option<Self> {
        match s {
            "mid" => Some(TargetPos::Midpoint),
            "q1" => Some(TargetPos::Quartile),
            _ => None,
        }
    }
}

fn gen_values(n: usize, regime: Regime, rng: &mut StdRng) -> Vec<u64> {
    match regime {
        Regime::Sparse => (0..n).map(|_| rng.gen_range(1..=(10 * n as u64))).collect(),
        Regime::Dense => (0..n)
            .map(|_| rng.gen_range(1..=(n as u64 / 2 + 1)))
            .collect(),
        Regime::EqualDenom => vec![50_000u64; n],
        Regime::Stdenom => {
            let denoms: &[u64] = &[5_000, 6_561, 10_000, 32_768, 65_536, 100_000, 262_144];
            (0..n)
                .map(|_| denoms[rng.gen_range(0..denoms.len())])
                .collect()
        }
    }
}

/// Target wall time per cell for the adaptive timing loop. Cells faster than
/// this take more iterations to average out noise; slow cells finish after one.
const CELL_BUDGET: Duration = Duration::from_millis(5);

/// Cap iterations so a pathologically fast call doesn't spin forever if the
/// clock misbehaves. 10M is unreachable for anything that does real work.
const MAX_ITERS: u64 = 10_000_000;

fn measure<F, R>(f: F) -> (u128, usize, bool)
where
    F: Fn() -> Option<R>,
{
    // One warmup pass to prime caches and branch predictors, discarded.
    drop(black_box(f()));

    PEAK.reset_peak_usage();
    let mut iters: u64 = 0;
    let last_is_some;
    let start = Instant::now();
    loop {
        let r = black_box(f());
        iters += 1;
        if start.elapsed() >= CELL_BUDGET || iters >= MAX_ITERS {
            last_is_some = r.is_some();
            break;
        }
    }
    let ns_per_iter = start.elapsed().as_nanos() / iters as u128;
    let peak = PEAK.peak_usage();
    (ns_per_iter, peak, last_is_some)
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let regimes = load_regimes();
    let targets = load_targets();
    let cfg = LookupConfig::default();
    let ns = load_usizes("BENCH_NS", DEFAULT_NS);
    let block_sizes = load_usizes("BENCH_BLOCK_SIZES", DEFAULT_BLOCK_SIZES);

    println!("regime\ttarget\tN\tmethod\tmedian_ns\tpeak_bytes\tresult_is_some");
    std::io::stdout().flush().ok();
    for &regime in &regimes {
        for &target_pos in &targets {
            for &n in &ns {
                let values = gen_values(n, regime, &mut rng);
                let mid = values.len() / 2;
                let positives = values[..mid].to_vec();
                let negatives = values[mid..].to_vec();
                let sum: u64 = values.iter().sum();
                let target = match target_pos {
                    TargetPos::Midpoint => (sum / 2) as i64,
                    TargetPos::Quartile => (sum / 4) as i64,
                };

                let (ns_s, peak_s, some_s) =
                    measure(|| log_w_signed_sasamoto(&positives, &negatives, target));
                println!(
                    "{}\t{}\t{}\tsasamoto\t{}\t{}\t{}",
                    regime.label(),
                    target_pos.label(),
                    n,
                    ns_s,
                    peak_s,
                    some_s,
                );
                std::io::stdout().flush().ok();

                for &k in &block_sizes {
                    let (ns_l, peak_l, some_l) = measure(|| {
                        log_lookup_w_signed_target_aware_with_config(
                            &positives, &negatives, target, k, &cfg,
                        )
                    });
                    println!(
                        "{}\t{}\t{}\tlookup_k{}\t{}\t{}\t{}",
                        regime.label(),
                        target_pos.label(),
                        n,
                        k,
                        ns_l,
                        peak_l,
                        some_l,
                    );
                    std::io::stdout().flush().ok();
                }
            }
        }
    }
}
