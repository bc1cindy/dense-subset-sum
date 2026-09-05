import dss


def test_w_brute_exact_small():
    r = dss.w_brute([3, 5, 8], [8, 8], 8)
    assert r["kind"] in ("exact", "lower_bound", "log_approx", "unknown")
    if r["kind"] in ("exact", "lower_bound"):
        assert isinstance(r["count"], int) and r["count"] >= 0


def test_radix_mappings_equal_denoms_counts():
    r = dss.radix_mappings([50000, 50000, 50000], 8)   # one value, k=1, m=3 -> 3! = 6
    assert r["kind"] == "diagnostic"                   # counts a different object; bounds nothing
    assert r["count"] == 6


def test_radix_mappings_absent_denomination_contributes_nothing():
    # 5512 = 5000 + 512, and no output carries 512, so the k:1 exchange has no subset to swap.
    assert dss.radix_mappings([5000, 5000, 5000, 5512], 8)["count"] == 6


def test_w_sasamoto_returns_log_or_unknown():
    r = dss.w_sasamoto([100000] * 12, [50000] * 20)
    assert r["kind"] in ("log_approx", "unknown")
    if r["kind"] == "log_approx":
        assert isinstance(r["log_w"], float)


def test_w_sparse_shape():
    r = dss.w_sparse([3, 5, 8], [8, 8], 8)
    assert set(r.keys()) == {"kind", "count", "log_w"}


def test_radix_mappings_large_count_not_truncated():
    r = dss.radix_mappings([100000] * 21, 8)
    assert r["kind"] == "diagnostic"
    # one value with multiplicity 21 -> 21!, which exceeds u64::MAX (~1.84e19); it must come back
    # whole, not wrapped
    assert r["count"] == 51090942171709440000
    assert r["count"] > 2**64


def test_amounts_outside_the_bitcoin_domain_are_rejected():
    import pytest

    for call in (
        lambda: dss.w_brute([2**63, 2**63], [1], 4),
        lambda: dss.w_sparse([1], [2**63, 2**63], 4),
        lambda: dss.w_count([2**63, 2**63], [1]),
        lambda: dss.w_sasamoto([2**63, 2**63], [1]),
        lambda: dss.radix_mappings([2**63, 2**63], 4),
        lambda: dss.per_coin_density([2**63, 2**63], [1]),
    ):
        with pytest.raises(ValueError):
            call()
