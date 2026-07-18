import dss

dense = dss.per_coin_density([1, 2, 3, 4, 5, 6, 7, 8], [18, 18])
sparse = dss.per_coin_density([1, 2, 4, 8, 16, 32], [21, 42])

assert set(dense) == {"kappa", "coins"}
assert dense["coins"] and all({"role", "index", "value", "log_w", "kappa_c"} <= set(c) for c in dense["coins"])
print("dense kappa:", dense["kappa"], "| out log_w:", [c["log_w"] for c in dense["coins"] if c["role"] == "out"])
print("sparse kappa:", sparse["kappa"], "| out log_w:", [c["log_w"] for c in sparse["coins"] if c["role"] == "out"])
print("ok")
