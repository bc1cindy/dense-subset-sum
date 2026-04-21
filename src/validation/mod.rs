//! Cross-validation of estimators against brute force and CJA mappings.

pub use density_sweep::{SubsetDensityPoint, print_subset_density_sweep, subset_density_sweep};
pub use per_coin::{
    CoinMeasurement, CoinRole, per_coin_measurements, per_coin_measurements_fee_aware,
    print_per_coin_measurements,
};
pub use sub_tx_estimates::{SubTxEstimate, estimate_sub_txs};

pub(super) fn exclude_values(full: &[u64], to_remove: &[u64]) -> Vec<u64> {
    let mut remaining = full.to_vec();
    for &val in to_remove {
        if let Some(pos) = remaining.iter().position(|&v| v == val) {
            remaining.swap_remove(pos);
        }
    }
    remaining
}

mod density_sweep;
mod per_coin;
mod sub_tx_estimates;
