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

pub use lookup::{dp_w, log_lookup_w, log_lookup_w_signed_target_aware, lookup_w, sumset_cap};
pub use radix::is_radix_like_in_base;
pub use regime::{kappa, kappa_c};
pub use sasamoto::{log_w_for_e_sat, log_w_signed_sasamoto};
pub use transaction::Transaction;

pub(crate) use sasamoto::gcd_slice;

/// One-sided Sasamoto threshold: below this |A|, the saddle approximation is
/// unreliable and the lookup/DP path is authoritative.
///
/// The signed (two-sided) probe uses a larger threshold (see
/// `SIGNED_SASAMOTO_THRESHOLD` in `validation::per_coin`) because its reliability
/// is indexed by `positives.len() + negatives.len()` (the whole ±multiset), not
/// by a single side's cardinality.
pub const SASAMOTO_MIN_N: usize = 20;
