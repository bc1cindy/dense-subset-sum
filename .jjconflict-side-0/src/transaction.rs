//! Bitcoin transaction reduced to input/output value lists (sat amounts).

use crate::Ambiguity;
use crate::compute;
use std::num::NonZeroUsize;

/// Inputs and outputs as `Vec<u64>` in satoshi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

impl Transaction {
    #[must_use]
    pub fn new(inputs: Vec<u64>, outputs: Vec<u64>) -> Self {
        Self { inputs, outputs }
    }

    #[must_use]
    pub fn input_sum(&self) -> u64 {
        self.inputs.iter().sum()
    }

    #[must_use]
    pub fn output_sum(&self) -> u64 {
        self.outputs.iter().sum()
    }

    #[must_use]
    pub fn fee(&self) -> i64 {
        let inp = i64::try_from(self.input_sum()).expect("input_sum ≤ MAX_MONEY < i64::MAX");
        let out = i64::try_from(self.output_sum()).expect("output_sum ≤ MAX_MONEY < i64::MAX");
        inp - out
    }

    /// Count of value entries (`inputs.len() + outputs.len()`), not byte size.
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }

    /// See [`crate::w_brute`].
    #[must_use]
    pub fn w_brute(&self, max_size: usize) -> Ambiguity {
        compute::w_brute(&self.inputs, &self.outputs, max_size)
    }

    /// See [`crate::w_sparse`].
    #[must_use]
    pub fn w_sparse(&self, max_size: usize, memory_budget: NonZeroUsize) -> Ambiguity {
        compute::w_sparse(&self.inputs, &self.outputs, max_size, memory_budget)
    }

    /// See [`crate::w_sasamoto`].
    #[must_use]
    pub fn w_sasamoto(&self) -> Ambiguity {
        compute::w_sasamoto(&self.inputs, &self.outputs)
    }

    /// See [`crate::radix_mappings`].
    #[must_use]
    pub fn radix_mappings(&self, max_size: usize) -> Ambiguity {
        compute::radix_mappings(&self.outputs, max_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_MEMORY_BUDGET;

    fn sample() -> Transaction {
        Transaction::new(vec![100, 200, 300], vec![150, 350, 100])
    }

    #[test]
    fn fee_matches_input_minus_output() {
        let tx = sample();
        assert_eq!(tx.input_sum(), 600);
        assert_eq!(tx.output_sum(), 600);
        assert_eq!(tx.fee(), 0);
    }

    #[test]
    fn fee_handles_negative() {
        let tx = Transaction::new(vec![100], vec![100, 50]);
        assert_eq!(tx.fee(), -50);
    }

    #[test]
    fn w_brute_method_matches_free_fn() {
        let tx = sample();
        assert_eq!(tx.w_brute(5), compute::w_brute(&tx.inputs, &tx.outputs, 5));
    }

    #[test]
    fn w_sparse_method_matches_free_fn() {
        let tx = sample();
        assert_eq!(
            tx.w_sparse(5, DEFAULT_MEMORY_BUDGET),
            compute::w_sparse(&tx.inputs, &tx.outputs, 5, DEFAULT_MEMORY_BUDGET),
        );
    }

    #[test]
    fn w_sasamoto_method_matches_free_fn() {
        let tx = sample();
        assert_eq!(
            tx.w_sasamoto(),
            compute::w_sasamoto(&tx.inputs, &tx.outputs)
        );
    }

    #[test]
    fn radix_mappings_method_matches_free_fn() {
        let tx = Transaction::new(vec![], vec![1000, 1000]);
        assert_eq!(
            tx.radix_mappings(6),
            compute::radix_mappings(&tx.outputs, 6)
        );
    }
}
