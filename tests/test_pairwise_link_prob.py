import dss

def test_single_in_single_out_is_deterministic():
    assert dss.pairwise_link_prob([3], [3]) == [[1.0]]

def test_two_in_one_out_both_linked():
    # [2,3] -> [5]: the only balanced sub-tx is {2,3}->{5}; both inputs link to the one output.
    assert dss.pairwise_link_prob([2, 3], [5]) == [[1.0], [1.0]]

def test_shape_matches_real_outputs_with_fee():
    # unbalanced (fee = 1): matrix has one row per input, one column per REAL output (fee dropped).
    m = dss.pairwise_link_prob([10, 10], [7, 12])  # fee = 1
    assert len(m) == 2 and all(len(row) == 2 for row in m)
    assert all(0.0 <= x <= 1.0 for row in m for x in row)

def test_returns_none_above_guard():
    big_in = [1] * 13
    big_out = [1] * 13  # 26 combined > 24
    assert dss.pairwise_link_prob(big_in, big_out) is None

def test_negative_fee_returns_none():
    assert dss.pairwise_link_prob([5], [9]) is None  # outputs exceed inputs
