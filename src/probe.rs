//! Coverage and balance probes over Sumset for the sub-transaction model's lower bound on indeterminacy.

use crate::sumset::Sumset;

/// `Cleared{k}` reports the smallest input degree where coverage held; bounded enumeration makes the answer conservative.
#[derive(Debug, PartialEq, Eq)]
pub enum ProbeResult {
    Cleared { at_fixed_degree: usize },
    Exhausted,
}

/// Stops at the smallest input degree clearing the threshold; bounded enumeration of inputs makes `Cleared` a conservative answer.
pub fn probe(
    intended_outputs: &[u64],
    other_inputs: &[u64],
    fixed_degree: usize,
    threshold: u8,
) -> ProbeResult {
    let outputs_sumset = Sumset::powerset(intended_outputs);
    let cap = fixed_degree.min(other_inputs.len());
    for k in 1..=cap {
        let inputs_sumset = Sumset::bounded(other_inputs, k);
        if outputs_sumset.covers(&inputs_sumset, threshold) {
            return ProbeResult::Cleared { at_fixed_degree: k };
        }
    }
    ProbeResult::Exhausted
}

/// Bounds the target side too; equivalent to `probe` when `target_degree == intended_outputs.len()`.
pub fn probe_bounded_target(
    intended_outputs: &[u64],
    other_inputs: &[u64],
    target_degree: usize,
    fixed_degree: usize,
    threshold: u8,
) -> ProbeResult {
    let outputs_sumset = Sumset::bounded(intended_outputs, target_degree);
    let cap = fixed_degree.min(other_inputs.len());
    for k in 1..=cap {
        let inputs_sumset = Sumset::bounded(other_inputs, k);
        if outputs_sumset.covers(&inputs_sumset, threshold) {
            return ProbeResult::Cleared { at_fixed_degree: k };
        }
    }
    ProbeResult::Exhausted
}

#[derive(Debug, PartialEq, Eq)]
pub struct TwoDirectionResult {
    pub a_to_b: ProbeResult,
    pub b_to_a: ProbeResult,
}

impl TwoDirectionResult {
    /// The conservative privacy claim only holds when both directions clear simultaneously.
    pub fn both_cleared(&self) -> bool {
        matches!(
            (&self.a_to_b, &self.b_to_a),
            (ProbeResult::Cleared { .. }, ProbeResult::Cleared { .. })
        )
    }
}

/// Runs `probe` in both directions; the conservative claim requires both to clear.
pub fn probe_two_direction(
    side_a: &[u64],
    side_b: &[u64],
    fixed_degree: usize,
    threshold: u8,
) -> TwoDirectionResult {
    TwoDirectionResult {
        a_to_b: probe(side_a, side_b, fixed_degree, threshold),
        b_to_a: probe(side_b, side_a, fixed_degree, threshold),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumset::Bound;

    #[test]
    fn alice_example_at_fixed_degree_two() {
        let alice_outputs = [10u64, 5, 4];
        let others_inputs = [8u64, 3, 6, 10];

        let bounded = Sumset::bounded(&others_inputs, 2);
        assert_eq!(bounded.bound(), Bound::LowerBound);

        for s in [0, 3, 6, 8, 9, 10, 11, 13, 14, 16, 18] {
            assert!(bounded.includes(s));
        }
        for s in [4, 5, 15, 19] {
            assert!(!bounded.includes(s));
        }

        let result = probe(&alice_outputs, &others_inputs, 2, 1);
        assert!(matches!(result, ProbeResult::Exhausted));
    }

    #[test]
    fn alice_example_at_higher_fixed_degrees() {
        let alice_outputs = [10u64, 5, 4];
        let others_inputs = [8u64, 3, 6, 10];

        let full = Sumset::bounded(&others_inputs, others_inputs.len());
        assert_eq!(full.bound(), Bound::Exact);

        // 3 + 6 + 10 = 19 unlocks at degree 3; 4, 5, 15 have no decomposition over {8, 3, 6, 10}.
        assert!(full.includes(19));
        assert!(!full.includes(4));
        assert!(!full.includes(5));
        assert!(!full.includes(15));

        for fixed_degree in 1..=others_inputs.len() {
            let result = probe(&alice_outputs, &others_inputs, fixed_degree, 1);
            assert_eq!(
                result,
                ProbeResult::Exhausted,
                "fixed_degree={}",
                fixed_degree
            );
        }
    }

    #[test]
    fn alice_like_example_clears_at_fixed_degree_one() {
        let alice_outputs = [3u64, 5];
        let others_inputs = [3u64, 5, 8];

        let result = probe(&alice_outputs, &others_inputs, 3, 1);
        assert_eq!(result, ProbeResult::Cleared { at_fixed_degree: 1 });
    }

    #[test]
    fn covers_singleton_match() {
        let user = Sumset::powerset(&[3]);
        let other = Sumset::bounded(&[3], 1);
        assert!(user.covers(&other, 1));
        assert!(!user.covers(&other, 2));
    }

    #[test]
    fn probe_clears_at_fixed_degree_one_when_singletons_cover() {
        let result = probe(&[5], &[5], 3, 1);
        assert_eq!(result, ProbeResult::Cleared { at_fixed_degree: 1 });
    }

    #[test]
    fn probe_exhausted_when_threshold_unreachable() {
        let result = probe(&[1], &[1], 1, 2);
        assert_eq!(result, ProbeResult::Exhausted);
    }

    #[test]
    fn probe_empty_other_side_is_exhausted() {
        let result = probe(&[3, 5], &[], 5, 1);
        assert_eq!(result, ProbeResult::Exhausted);
    }

    #[test]
    fn probe_bounded_target_with_full_target_degree_matches_probe() {
        let outputs = [3u64, 5, 7];
        let inputs = [1u64, 2, 4, 6, 8];

        let bounded = probe_bounded_target(&outputs, &inputs, outputs.len(), 3, 1);
        let regular = probe(&outputs, &inputs, 3, 1);
        assert_eq!(bounded, regular);
    }

    #[test]
    fn probe_bounded_target_smaller_target_degree_can_clear_when_full_does_not() {
        let outputs = [1u64, 2, 3];
        let inputs = [1u64, 2];

        assert_eq!(
            probe(&outputs, &inputs, inputs.len(), 1),
            ProbeResult::Exhausted
        );
        assert!(matches!(
            probe_bounded_target(&outputs, &inputs, 1, inputs.len(), 1),
            ProbeResult::Cleared { .. }
        ));
    }

    #[test]
    fn two_direction_both_cleared_for_symmetric_inputs_outputs() {
        let result = probe_two_direction(&[1u64, 2], &[1u64, 2], 2, 1);
        assert!(result.both_cleared());
    }

    #[test]
    fn two_direction_asymmetric_one_clears_other_exhausts() {
        let result = probe_two_direction(&[1u64, 2], &[1u64, 2, 3, 4], 4, 1);
        assert!(matches!(result.a_to_b, ProbeResult::Cleared { .. }));
        assert_eq!(result.b_to_a, ProbeResult::Exhausted);
        assert!(!result.both_cleared());
    }

    #[test]
    fn two_direction_alice_example_both_exhaust() {
        let result = probe_two_direction(&[10u64, 5, 4], &[8u64, 3, 6, 10], 2, 1);
        assert!(!result.both_cleared());
        assert_eq!(result.a_to_b, ProbeResult::Exhausted);
    }
}
