use crate::Transaction;

/// Maurer et al. Fig 2 (mappings.md:189): inputs {21,12,36,28}, outputs {25,8,50,14}.
#[must_use]
pub fn maurer_fig2() -> Transaction {
    Transaction::new(vec![21, 12, 36, 28], vec![25, 8, 50, 14])
}

/// Whirlpool-style equal-amount edge case (κ_c = 0).
#[must_use]
pub fn equal_denominations() -> Transaction {
    Transaction::new(
        vec![100_000, 100_000, 100_000, 50_000, 50_000],
        vec![100_000, 100_000, 100_000, 50_000, 50_000],
    )
}
