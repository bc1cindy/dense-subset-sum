//! Graded subset-sum sumset (per-degree buckets) with saturating u32 counts.

mod budget;
mod graded;
#[cfg(test)]
mod test_oracle;
mod types;

pub use budget::GradedSumsetBudget;
pub use graded::{GradedSumset, GradedSumsetBuilder};
pub use types::{Bound, Count};
