//! CoinJoin transaction in satoshis.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

impl Transaction {
    pub fn new(inputs: Vec<u64>, outputs: Vec<u64>) -> Self {
        Self { inputs, outputs }
    }

    pub fn input_sum(&self) -> u64 {
        self.inputs.iter().sum()
    }

    pub fn output_sum(&self) -> u64 {
        self.outputs.iter().sum()
    }

    pub fn fee(&self) -> Option<u64> {
        self.input_sum().checked_sub(self.output_sum())
    }

    pub fn n_coins(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }
}
