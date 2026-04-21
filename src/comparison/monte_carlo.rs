//! Monte Carlo ground truth for `W(E)` when N > 25 (2^N enumeration infeasible).

use std::collections::HashMap;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::core::{CompareMode, ComparisonReport, build_report_from_counts};

#[allow(clippy::too_many_arguments)]
pub fn compare_monte_carlo(
    a: &[u64],
    min_count: u64,
    lookup_k: usize,
    dp_max: usize,
    label: &str,
    n_samples: u64,
    timeout_ms: u64,
    seed: u64,
) -> ComparisonReport {
    let n = a.len();
    assert!(n > 0, "compare_monte_carlo: empty input");
    assert!(n < 1024, "compare_monte_carlo: N={} impractical", n);
    assert!(
        n_samples > 0,
        "compare_monte_carlo: n_samples=0 produces no estimate; \
         pass timeout_ms=0 + n_samples=u64::MAX for a timeout-only run"
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let mut counts: HashMap<u64, u64> = HashMap::new();
    let mut samples_drawn: u64 = 0;

    let start = Instant::now();
    // timeout_ms=0 means "no wall-clock limit"; rely on n_samples only.
    let timeout = (timeout_ms > 0).then(|| std::time::Duration::from_millis(timeout_ms));

    while samples_drawn < n_samples {
        if let Some(t) = timeout
            && samples_drawn.is_multiple_of(MC_TIMEOUT_CHECK_INTERVAL)
            && start.elapsed() >= t
        {
            break;
        }
        let mut sum: u64 = 0;
        for &v in a {
            if rng.r#gen::<bool>() {
                sum = sum.saturating_add(v);
            }
        }
        *counts.entry(sum).or_insert(0) += 1;
        samples_drawn += 1;
    }

    let timed_out = samples_drawn < n_samples;
    let scale = if samples_drawn == 0 {
        1.0
    } else {
        2.0_f64.powi(n as i32) / samples_drawn as f64
    };

    build_report_from_counts(
        a,
        &counts,
        scale,
        min_count,
        lookup_k,
        dp_max,
        label,
        CompareMode::MonteCarlo {
            samples_requested: n_samples,
            samples_drawn,
            timed_out,
        },
    )
}

/// Sample-count interval between timeout checks in `compare_monte_carlo`.
const MC_TIMEOUT_CHECK_INTERVAL: u64 = 1024;
