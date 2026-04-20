//! Change decomposition — Maurer §5 output splitting.
//!
//! Separate from `estimator` because change decomposition is a *search* over candidate
//! output-side splits ranked by mean signed log₂W.

use crate::{SignedMethod, Transaction, validation};

/// Maurer §5 output splitting. Returns `(best_split, best_mean_log2_w_signed)`.
///
/// Candidates: single piece, N equal pieces, greedy-match existing denominations,
/// and power-of-two decomposition (low Hamming weight).
pub fn suggest_output_split(
    tx: &Transaction,
    change_amount: u64,
    max_pieces: usize,
    lookup_k: usize,
    method: SignedMethod,
) -> (Vec<u64>, f64) {
    if change_amount == 0 {
        return (vec![], f64::NEG_INFINITY);
    }
    let max_pieces = max_pieces.clamp(1, 8);

    let mut candidates: Vec<Vec<u64>> = Vec::new();
    candidates.push(vec![change_amount]);

    for n in 2..=max_pieces {
        let base = change_amount / n as u64;
        if base == 0 {
            continue;
        }
        let remainder = change_amount - base * n as u64;
        let mut split = vec![base; n];
        split[0] += remainder;
        candidates.push(split);
    }

    // Greedy largest-first over existing tx denominations.
    let mut existing_denoms: Vec<u64> = tx.outputs.clone();
    existing_denoms.sort_unstable_by(|a, b| b.cmp(a));
    existing_denoms.dedup();
    {
        let mut remaining = change_amount;
        let mut split = Vec::new();
        for &d in &existing_denoms {
            if d == 0 {
                continue;
            }
            while remaining >= d && split.len() < max_pieces {
                split.push(d);
                remaining -= d;
            }
        }
        if remaining > 0 && split.len() < max_pieces {
            split.push(remaining);
        }
        if (remaining == 0 || split.len() <= max_pieces)
            && split.iter().sum::<u64>() == change_amount
        {
            candidates.push(split);
        }
    }

    // Power-of-two decomposition: low Hamming weight in base 2.
    {
        let mut remaining = change_amount;
        let mut split = Vec::new();
        while remaining > 0 && split.len() < max_pieces {
            let highest_bit = 1u64 << (63 - remaining.leading_zeros());
            split.push(highest_bit);
            remaining -= highest_bit;
        }
        if remaining > 0 && !split.is_empty() {
            *split.last_mut().unwrap() += remaining;
        }
        if split.iter().sum::<u64>() == change_amount {
            candidates.push(split);
        }
    }

    let mut best_split = vec![change_amount];
    let mut best_mean = f64::NEG_INFINITY;

    for candidate in &candidates {
        let mut test_outputs = tx.outputs.clone();
        test_outputs.extend_from_slice(candidate);
        let test_tx = Transaction::new(tx.inputs.clone(), test_outputs);
        let measurements = validation::per_coin_measurements(&test_tx, lookup_k, method);
        let ln2 = std::f64::consts::LN_2;
        let reachable: Vec<f64> = measurements
            .iter()
            .filter_map(|c| c.log_w_signed)
            .map(|v| v / ln2)
            .collect();
        let mean = if reachable.is_empty() {
            f64::NEG_INFINITY
        } else {
            reachable.iter().sum::<f64>() / reachable.len() as f64
        };
        if mean > best_mean {
            best_mean = mean;
            best_split = candidate.clone();
        }
    }

    (best_split, best_mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggest_output_split_preserves_total() {
        let tx = Transaction::new(vec![100, 200, 300], vec![100, 200, 250]);
        let change = 50u64;
        let (split, mean) = suggest_output_split(&tx, change, 4, 3, SignedMethod::Lookup);
        assert_eq!(split.iter().sum::<u64>(), change);
        assert!(mean.is_finite() || split.len() == 1);
    }

    #[test]
    fn test_suggest_output_split_zero_change() {
        let tx = Transaction::new(vec![100], vec![100]);
        let (split, _) = suggest_output_split(&tx, 0, 4, 3, SignedMethod::Lookup);
        assert!(split.is_empty());
    }

    #[test]
    fn test_suggest_output_split_picks_best() {
        let tx = Transaction::new(vec![100, 100, 100, 100], vec![100, 100, 100, 50]);
        let (split, best_mean) = suggest_output_split(&tx, 50, 4, 3, SignedMethod::Lookup);
        assert_eq!(split.iter().sum::<u64>(), 50);
        let single_tx = Transaction::new(tx.inputs.clone(), {
            let mut o = tx.outputs.clone();
            o.push(50);
            o
        });
        let single_measurements =
            validation::per_coin_measurements(&single_tx, 3, SignedMethod::Lookup);
        let ln2 = std::f64::consts::LN_2;
        let r: Vec<f64> = single_measurements
            .iter()
            .filter_map(|c| c.log_w_signed)
            .map(|v| v / ln2)
            .collect();
        let single_mean = if r.is_empty() {
            f64::NEG_INFINITY
        } else {
            r.iter().sum::<f64>() / r.len() as f64
        };
        assert!(best_mean >= single_mean - 1e-9);
    }
}
