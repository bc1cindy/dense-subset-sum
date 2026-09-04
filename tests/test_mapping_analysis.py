import re
from itertools import combinations, permutations
from math import log2

import dss
import pytest


def test_complete_analysis_has_a_stable_schema():
    assert dss.mapping_analysis([3], [3]) == {
        "status": "complete",
        "n_non_derived": 1,
        "entropy": 0.0,
        "deterministic_links": [(0, 0)],
    }


def test_fee_output_is_not_exposed_as_a_deterministic_link():
    result = dss.mapping_analysis([10], [9])
    assert result["status"] == "complete"
    assert all(output == 0 for _, output in result["deterministic_links"])


def test_size_guard_and_invalid_value_conservation_are_distinct():
    guarded = dss.mapping_analysis([1] * 13, [1] * 13)
    assert guarded == {"status": "refused", "reason": "size_guard"}

    invalid = dss.mapping_analysis([5], [9])
    assert invalid == {"status": "invalid", "reason": "negative_fee"}


def test_amounts_outside_the_bitcoin_domain_are_rejected_without_panicking():
    max_money = 2_100_000_000_000_000
    assert dss.mapping_analysis([max_money, 1], [max_money]) == {
        "status": "invalid",
        "reason": "amount_out_of_range",
    }
    assert dss.mapping_analysis([2**64 - 1, 1], [0]) == {
        "status": "invalid",
        "reason": "amount_out_of_range",
    }
    assert dss.pairwise_link_prob([2**63], [0]) is None


def test_expired_budget_identifies_the_interrupted_phase():
    result = dss.mapping_analysis([1, 1], [1, 1], budget_ms=0)
    assert result == {"status": "refused", "reason": "enumeration_deadline"}


def test_dense_fast_path_does_not_claim_an_exact_mapping_count():
    inputs = [4_000_000] * 12
    outputs = [20_000] * 8 + [2_097_152] * 8 + [5_000_000] * 4
    result = dss.mapping_analysis(inputs, outputs, budget_ms=100)
    assert result == {"status": "dense_fast_path"}
    assert dss.pairwise_link_prob(inputs, outputs, budget_ms=100) is None


def test_build_metadata_is_machine_readable():
    assert dss.__version__ == "0.1.0"
    assert dss.__rev__ is None or re.fullmatch(r"[0-9a-f]{40}", dss.__rev__)


def test_published_result_revision_gate():
    if dss.__rev__ is None:
        with pytest.raises(RuntimeError, match="published results require"):
            dss.require_build_revision()
    else:
        assert dss.require_build_revision(dss.__rev__) == dss.__rev__
        different_revision = ("0" if dss.__rev__[0] != "0" else "1") + dss.__rev__[1:]
        with pytest.raises(RuntimeError, match="does not match"):
            dss.require_build_revision(different_revision)


def _partitions(items):
    if not items:
        yield ()
        return
    first, *rest = items
    for partition in _partitions(rest):
        yield ((first,), *partition)
        for index in range(len(partition)):
            yield (*partition[:index], (first, *partition[index]), *partition[index + 1:])


def _canonical_mapping(blocks):
    return tuple(sorted((tuple(sorted(inputs)), tuple(sorted(outputs)))
                        for inputs, outputs in blocks))


def _independent_non_derived_mappings(inputs, outputs):
    mappings = set()
    for input_partition in _partitions(tuple(range(len(inputs)))):
        for output_partition in _partitions(tuple(range(len(outputs)))):
            if len(input_partition) != len(output_partition):
                continue
            for arranged_outputs in permutations(output_partition):
                blocks = tuple(zip(input_partition, arranged_outputs))
                if all(sum(inputs[i] for i in input_block)
                       == sum(outputs[o] for o in output_block)
                       for input_block, output_block in blocks):
                    mappings.add(_canonical_mapping(blocks))

    non_derived = set(mappings)
    for finer in mappings:
        if len(finer) < 2:
            continue
        for left, right in combinations(range(len(finer)), 2):
            merged = [block for index, block in enumerate(finer)
                      if index not in (left, right)]
            merged.append((finer[left][0] + finer[right][0],
                           finer[left][1] + finer[right][1]))
            non_derived.discard(_canonical_mapping(merged))
    return non_derived


def _independent_pairwise_matrix(inputs, outputs, mappings):
    matrix = [[0.0 for _ in outputs] for _ in inputs]
    if not mappings:
        return matrix
    for input_index in range(len(inputs)):
        for output_index in range(len(outputs)):
            hits = sum(any(input_index in input_block and output_index in output_block
                           for input_block, output_block in mapping)
                       for mapping in mappings)
            matrix[input_index][output_index] = hits / len(mappings)
    return matrix


@pytest.mark.parametrize("inputs,outputs", [
    ([1], [1]),
    ([1, 2], [1, 2]),
    ([1, 2, 3], [1, 2, 3]),
    ([1, 2, 4], [3, 4]),
    ([1, 3, 5], [4, 5]),
])
def test_mapping_diagnostics_match_an_independent_exhaustive_oracle(inputs, outputs):
    mappings = _independent_non_derived_mappings(inputs, outputs)
    expected_matrix = _independent_pairwise_matrix(inputs, outputs, mappings)
    analysis = dss.mapping_analysis(inputs, outputs)
    matrix = dss.pairwise_link_prob(inputs, outputs)

    assert analysis["status"] == "complete"
    assert analysis["n_non_derived"] == len(mappings)
    assert analysis["entropy"] == pytest.approx(log2(len(mappings)))
    assert len(matrix) == len(expected_matrix)
    for actual_row, expected_row in zip(matrix, expected_matrix):
        assert actual_row == pytest.approx(expected_row)
    expected_certain = [(i, o) for i, row in enumerate(expected_matrix)
                        for o, probability in enumerate(row) if probability == 1.0]
    assert analysis["deterministic_links"] == expected_certain
