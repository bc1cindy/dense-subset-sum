//! W(E) subset-sum counting for privacy analysis.
//!
//! W(E) = number of input subsets that sum to E. Provides exact (DP),
//! lookup lower-bound, and Sasamoto asymptotic estimators.

pub mod regime;
pub mod sasamoto;
pub mod stats;

pub use regime::{density_regime, kappa, kappa_c};
pub use sasamoto::{log_w_for_e, log_w_for_e_sat};
pub use transaction::Transaction;

mod transaction;
