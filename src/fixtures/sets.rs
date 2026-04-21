//! General Transaction fixtures + bare value-set fixtures (synthetic, paper examples).

use crate::Transaction;

/// Maurer et al. Fig 2: 2 mappings (1 non-derived + 1 derived).
pub fn maurer_fig2() -> Transaction {
    Transaction::new(vec![21, 12, 36, 28], vec![25, 8, 50, 14])
}

pub fn equal_denominations() -> Transaction {
    Transaction::new(
        vec![100_000, 100_000, 100_000, 50_000, 50_000],
        vec![100_000, 100_000, 100_000, 50_000, 50_000],
    )
}

/// Values > 10^12 sat — exercises f64 precision.
pub fn large_values() -> Transaction {
    Transaction::new(
        vec![1_500_000_000_000, 800_000_000_000, 300_000_000_000],
        vec![1_200_000_000_000, 900_000_000_000, 500_000_000_000],
    )
}

pub fn with_fee() -> Transaction {
    Transaction::new(
        vec![50_000, 30_000, 20_000, 10_000],
        vec![45_000, 25_000, 19_000, 20_000],
    )
}

pub fn sequential_small() -> Vec<u64> {
    (1..=8).collect()
}

pub fn sequential_medium() -> Vec<u64> {
    (1..=16).collect()
}

pub fn sequential_large() -> Vec<u64> {
    (1..=20).collect()
}

pub fn powers_of_two() -> Vec<u64> {
    (0..10).map(|i| 1u64 << i).collect()
}

pub fn wasabi_denominations() -> Vec<u64> {
    vec![
        5_000, 5_000, 5_000, 10_000, 10_000, 10_000, 50_000, 50_000, 100_000, 100_000, 100_000,
        100_000,
    ]
}

pub fn mixed_radix_arbitrary() -> Vec<u64> {
    vec![1, 2, 4, 8, 16, 32, 137, 293, 571, 823, 1049, 1511]
}

pub fn single_outlier() -> Vec<u64> {
    vec![100, 100, 100, 100, 100, 100, 100, 100, 100, 1_000_000]
}

pub fn all_equal() -> Vec<u64> {
    vec![1_000; 15]
}

/// Near κ_c phase transition: N=16, max≈2^16 ⇒ κ≈1.0.
pub fn near_boundary() -> Vec<u64> {
    vec![
        3421, 58102, 12847, 45231, 29654, 7813, 51090, 38476, 62519, 8734, 21567, 44983, 16205,
        53871, 37042, 65001,
    ]
}

/// First 8 inputs of Wasabi 2 tx 54818554f0f69ad6.
pub fn real_wasabi_small() -> Vec<u64> {
    vec![5_000, 5_000, 5_000, 5_000, 10_000, 10_000, 10_000, 50_000]
}
