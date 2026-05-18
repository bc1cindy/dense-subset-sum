pub mod ambiguity;
pub mod compute;
pub mod count;
pub mod loss;

pub use loss::LossError;

pub use ambiguity::Ambiguity;
pub use compute::{DEFAULT_MEMORY_BUDGET, KNEE, radix_mappings, w_brute, w_sasamoto, w_sparse};

pub use count::density_regime::{
    Bracket, Interval, L, MAX_MONEY, Regime, kappa, kappa_c, regime_at_l, worst_case_kappa,
};
pub use count::companion::{SignedError, log_w_signed, sasamoto_approx, sasamoto_approx_m};
pub use count::denoms::{
    binary_denoms_in_range, decimal_denoms_in_range, is_standard_denom, multiples_in_range,
    powers_in_range, standard_denoms_in_range, ternary_denoms_in_range,
};
pub use count::oracle::{
    BruteError, DpError, brute_force_w, brute_force_w_restricted, dp_w, dp_w_restricted,
};
pub use count::radix::{
    DEFAULT_DUST_SATS, DEFAULT_MAX_COMBINATION_SIZE, DEFAULT_MAX_COMBINATION_VALUE_SATS,
    DEFAULT_MAX_DENOM_SATS, DEFAULT_MIN_DENOM_SATS, DEFAULT_MIN_DIFF, DEFAULT_MIN_DIFF_RATIO,
    DUST_FEERATE_SAT_PER_VBYTE, EXPECTED_FEERATE_SAT_PER_VBYTE, MAX_SATS, P2WPKH_OUTPUT_VBYTES,
    approximate_from_below, dust_at_feerate, factorial, radix_decompose, radix_gaps,
    radix_gaps_per_k, radix_mapping_count, radix_relative_gaps, radix_relative_gaps_per_k,
    radix_sumset, radix_sumset_counts, radix_sumsets_up_to, sumset_density,
};
pub use count::sasamoto::{log_w_for_e_sat, log_w_for_m_e_sat, m_sqrt_over_2, n_c};
pub use count::shape::{InputShape, radix_density};
pub use count::sparse_conv::{Field, Goldilocks};
pub use count::sumset::{Bound, Count, GradedSumset, GradedSumsetBudget, GradedSumsetBuilder};
