//! W(E) subset-sum counting for privacy analysis.
//!
//! W(E) = number of input subsets that sum to E. This commit introduces
//! the Sasamoto asymptotic estimator; exact (DP) and lookup lower-bound
//! paths follow in subsequent commits.

pub mod regime;
pub mod sasamoto;
pub mod stats;

pub use regime::{density_regime, kappa, kappa_c};
pub use sasamoto::{log_w_for_e, log_w_for_e_sat, log_w_signed_sasamoto, n_c};
pub use transaction::Transaction;

/// One-sided Sasamoto threshold: below this |A|, the saddle approximation is
/// unreliable and the lookup/DP path is authoritative.
pub const SASAMOTO_MIN_N: usize = 20;

mod transaction;
