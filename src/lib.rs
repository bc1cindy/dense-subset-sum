//! Crate skeleton shared across subset-sum estimators and harnesses.
//!
//! Provides the [`Transaction`] vocabulary (inputs + outputs) and the
//! correlation/summary helpers in [`stats`]. Analysis modules are added
//! in subsequent commits.

pub mod stats;

pub use transaction::Transaction;

mod transaction;
