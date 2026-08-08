import dss


def test_w_brute_exact_small():
    r = dss.w_brute([3, 5, 8], [8, 8], 8)
    assert r["kind"] in ("exact", "lower_bound", "log_approx", "unknown")
    if r["kind"] in ("exact", "lower_bound"):
        assert isinstance(r["count"], int) and r["count"] >= 0


def test_radix_mappings_equal_denoms_counts():
    r = dss.radix_mappings([50000, 50000, 50000], 8)   # equal denoms -> Σ k·m! mapping count
    assert r["kind"] in ("exact", "lower_bound")
    assert r["count"] is not None and r["count"] >= 1


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
    assert r["kind"] == "exact"
    # this count exceeds u64::MAX (~1.84e19); it must come back exact, not wrapped
    assert r["count"] == 1072909785605898240000
    assert r["count"] > 2**64
