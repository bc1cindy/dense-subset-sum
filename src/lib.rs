//! W(E) subset-sum counting for privacy analysis.
//!
//! W(E) = number of input subsets that sum to E. Provides exact (DP),
//! lookup lower-bound, and Sasamoto asymptotic estimators.

pub mod lookup;
pub mod regime;
pub mod sasamoto;
pub mod stats;

pub use lookup::{
    DEFAULT_MAX_ENTRIES, LookupConfig, brute_force_w, dp_w, dp_w_restricted, log_dp_w,
    log_dp_w_restricted, log_lookup_w, log_lookup_w_with_config, lookup_w, lookup_w_with_config,
    sumset_size_with_config,
};
pub use regime::{density_regime, kappa, kappa_c};
pub use sasamoto::{log_w_for_e, log_w_for_e_sat, n_c};
pub use transaction::Transaction;

mod transaction;
