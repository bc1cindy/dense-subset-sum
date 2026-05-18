//! CJA empirical UTXO distribution sampling + synthetic coinjoin builder.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::Transaction;

/// CJA empirical UTXO distribution. Fetch via `scripts/fetch_cja_distribution.sh` (~353 MB, LFS).
pub struct EmpiricalDistribution {
    cdf: Vec<(u64, f64)>,
}

impl EmpiricalDistribution {
    /// O(log N) per sample via binary search on the monotone CDF (23M points in CJA).
    pub fn sample(&self, rng: &mut impl Rng) -> u64 {
        let p: f64 = rng.r#gen();
        let idx = self.cdf.partition_point(|&(_, cp)| cp < p);
        if idx >= self.cdf.len() {
            return self
                .cdf
                .last()
                .expect("cdf non-empty (checked in from_cja_bin)")
                .0;
        }
        let (val, cum_p) = self.cdf[idx];
        let (prev_val, prev_p) = if idx == 0 {
            (0u64, 0.0)
        } else {
            self.cdf[idx - 1]
        };
        let bucket_frac = if cum_p > prev_p {
            (p - prev_p) / (cum_p - prev_p)
        } else {
            0.5
        };
        let sampled = prev_val + ((val - prev_val) as f64 * bucket_frac) as u64;
        sampled.max(1)
    }

    pub fn random_set(&self, n: usize, seed: u64) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| self.sample(&mut rng)).collect()
    }

    pub const DEFAULT_CJA_PATH: &'static str = "testdata/cja_distribution.bin";

    /// Returns `None` when the 353 MB binary is absent, so tests can skip cleanly.
    pub fn try_load_default_cja() -> Option<Self> {
        let path = std::path::Path::new(Self::DEFAULT_CJA_PATH);
        if path.exists() {
            Self::from_cja_bin(path).ok()
        } else {
            None
        }
    }

    pub fn from_cja_bin<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        #[derive(serde::Deserialize)]
        struct CjaDistribution {
            cumulative_normalized: Vec<(u64, f64)>,
        }
        let file = std::fs::File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);
        let cja: CjaDistribution = rmp_serde::from_read(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if cja.cumulative_normalized.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "CJA distribution.bin contained an empty CDF",
            ));
        }
        Ok(Self {
            cdf: cja.cumulative_normalized,
        })
    }

    pub fn cdf_len(&self) -> usize {
        self.cdf.len()
    }
}

/// Each participant: `inputs_per_participant` inputs + 2 outputs (random + change).
pub fn random_coinjoin(
    dist: &EmpiricalDistribution,
    n_participants: usize,
    inputs_per_participant: usize,
    seed: u64,
) -> Transaction {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_inputs = Vec::new();
    let mut all_outputs = Vec::new();

    for _ in 0..n_participants {
        let participant_inputs: Vec<u64> = (0..inputs_per_participant)
            .map(|_| dist.sample(&mut rng))
            .collect();
        let input_sum: u64 = participant_inputs.iter().sum();

        let output1 = rng.r#gen_range(1..input_sum);
        let output2 = input_sum - output1;

        all_inputs.extend(participant_inputs);
        all_outputs.push(output1);
        all_outputs.push(output2);
    }

    Transaction::new(all_inputs, all_outputs)
}
