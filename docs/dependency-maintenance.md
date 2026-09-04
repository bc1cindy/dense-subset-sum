# Dependency maintenance

## `nom` 2.2.1

`nom` 2.2.1 is not a direct dependency. It is introduced by the Git-pinned
`coinjoin_analyzer` revision used as the independent CJA mapping implementation. Cargo reports that
this version contains constructs that a future Rust release will reject.

The dependency must not be replaced blindly because its mapping behavior is part of the diagnostic
baseline. The migration is complete only when all of the following hold:

1. `coinjoin_analyzer` upgrades or removes its legacy `nom` dependency.
2. DSS pins a reviewed upstream revision rather than patching a transitive parser locally.
3. The mapping differential and property tests pass unchanged.
4. The decluster exact-oracle audit is regenerated and its result delta is explained.
5. `cargo report future-incompatibilities` no longer names `nom` 2.2.1.

Until then, CI should retain the warning and the exact `coinjoin_analyzer` revision in `Cargo.lock`.
