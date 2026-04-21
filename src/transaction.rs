/// A representation of Bitcoin transactions abstracting away everything but input and output values.
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

    pub fn fee(&self) -> i64 {
        self.input_sum() as i64 - (self.output_sum() as i64)
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }
}
