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
pub mod validation;

pub use lookup::{dp_w, log_lookup_w_signed_target_aware, lookup_w, sumset_cap};
pub use regime::{kappa, kappa_c};
pub use sasamoto::{log_w_for_e_sat, log_w_signed_sasamoto};
pub use transaction::Transaction;

pub(crate) use sasamoto::gcd_slice;
