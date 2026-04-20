//! W(E) subset-sum counting for privacy analysis.
//!
//! W(E) = number of input subsets that sum to E. Provides exact (DP),
//! lookup lower-bound, and Sasamoto asymptotic estimators.

pub mod change_split;
pub mod comparison;
pub mod dense_region;
pub mod fixtures;
pub mod lookup;
pub mod loss;
pub mod mappings;
pub mod radix;
pub mod regime;
pub mod sasamoto;
pub mod stats;
mod transaction;
pub mod validation;

pub use dense_region::find_dense_region;
pub use lookup::{
    DEFAULT_MAX_ENTRIES, LookupConfig, brute_force_w, dp_w, log_dp_w, log_lookup_w,
    log_lookup_w_signed_target_aware, log_lookup_w_signed_target_aware_with_config,
    log_lookup_w_with_config, log_w_signed_adaptive, log_w_signed_adaptive_with_config, lookup_w,
    lookup_w_with_config,
};
pub use radix::{
    RADIX_BASES, arbitrary_distinctness_log2, best_radix_base, coverage_bonus_log2,
    denomination_multiplicities, denomination_reward_log2, distinguish_coins,
    is_radix_like_any_base, is_radix_like_in_base,
};
pub use regime::{density_regime, kappa, kappa_c};
pub use sasamoto::{log_w_for_e, log_w_for_e_sat, log_w_signed_sasamoto};
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
