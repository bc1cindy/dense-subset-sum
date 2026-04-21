//! W(E) subset-sum counting for privacy analysis.
//!
//! W(E) = number of input subsets that sum to E. Provides exact (DP),
//! lookup lower-bound, and Sasamoto asymptotic estimators.

pub mod dense_region;
pub mod empirical_regime;
pub mod lookup;
pub mod penalty;
pub mod radix;
pub mod regime;
pub mod sasamoto;
pub mod stats;

pub use dense_region::find_dense_region;
pub use empirical_regime::{EmpiricalRegime, empirical_regime};
pub use lookup::{
    DEFAULT_MAX_ENTRIES, LookupConfig, SignedMethod, brute_force_w, dp_w, dp_w_restricted,
    log_dp_w, log_dp_w_restricted, log_lookup_w, log_lookup_w_signed_target_aware,
    log_lookup_w_signed_target_aware_with_config, log_lookup_w_with_config, log_w_signed,
    log_w_signed_with_config, lookup_w, lookup_w_with_config, sumset_size_with_config,
};
pub use penalty::{cluster_penalty_log, global_nc_over_n, m_sqrt_over_2, per_input_penalty_log};
pub use radix::{DISTINGUISHED, is_distinguished};
pub use regime::{density_regime, kappa, kappa_c};
pub use sasamoto::{log_w_for_e, log_w_for_e_sat, log_w_signed_sasamoto, n_c};
pub use transaction::Transaction;

pub(crate) use sasamoto::gcd_slice;

/// One-sided Sasamoto threshold: below this |A|, the saddle approximation is
/// unreliable and the lookup/DP path is authoritative.
pub const SASAMOTO_MIN_N: usize = 20;

mod transaction;
