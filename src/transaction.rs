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

    pub fn fee(&self) -> u64 {
        self.input_sum() - self.output_sum()
    }

    pub fn len(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.outputs.is_empty()
    }
}
