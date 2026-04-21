//! Crate skeleton shared across subset-sum estimators and harnesses.
//!
//! Provides the [`Transaction`] vocabulary (inputs + outputs), the
//! correlation/summary helpers in [`stats`], and the κ/κ_c density
//! regime classifier. Estimators are added in subsequent commits.

pub mod regime;
pub mod stats;

pub use regime::{density_regime, kappa, kappa_c};
pub use transaction::Transaction;

mod transaction;
