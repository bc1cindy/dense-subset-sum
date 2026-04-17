//! Foundational types and shared helpers for W(E) subset-sum analysis.

pub mod comparison;
pub mod dense_region;
pub mod fixtures;
pub mod lookup;
pub mod mappings;
pub mod radix;
pub mod regime;
pub mod sasamoto;
pub mod stats;
mod transaction;

pub use lookup::{dp_w, lookup_w};
pub use regime::kappa;
pub use sasamoto::log_w_for_e_sat;
pub use transaction::Transaction;

pub(crate) use sasamoto::gcd_slice;
