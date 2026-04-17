//! Transaction and value-set fixtures for tests, benchmarks, and the CLI.
//!
//! Organized by origin:
//! - `sets`: synthetic / paper-derived value sets and small general Transactions.
//! - `wasabi2_positive`: 30 manually-bucketed Wasabi 2 CoinJoins (positive samples).
//! - `wasabi2_false`: Wasabi 2 lookalikes (negative samples, consolidations + stdenom).

mod sets;
mod wasabi2_false;
mod wasabi2_positive;

pub use sets::*;
pub use wasabi2_false::*;
pub use wasabi2_positive::*;

pub fn all_comparison_sets() -> Vec<(&'static str, Vec<u64>)> {
    let mut sets = vec![
        ("sequential_small_N8", sequential_small()),
        ("sequential_medium_N16", sequential_medium()),
        ("sequential_large_N20", sequential_large()),
        ("powers_of_two_N10", powers_of_two()),
        ("wasabi_denominations_N12", wasabi_denominations()),
        ("mixed_radix_N12", mixed_radix_arbitrary()),
        ("single_outlier_N10", single_outlier()),
        ("all_equal_N15", all_equal()),
        ("near_boundary_N16", near_boundary()),
        ("real_wasabi_small_N8", real_wasabi_small()),
    ];
    for (label, tx) in all_wasabi2_false_cjtxs() {
        sets.push((label, tx.inputs));
    }
    sets
}
