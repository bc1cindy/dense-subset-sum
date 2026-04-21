# dense-subset-sum

A tool to measure **how private a CoinJoin transaction is**.

Given the inputs and outputs of a Bitcoin CoinJoin, it counts (or approximates) how many alternative ways an observer could plausibly match inputs to outputs. The count itself is the output; interpretation of what counts as "enough" is left to the caller.

You can use it to:

- Measure subset-sum density (`W(E)`, `κ`, signed log₂W) on an existing CoinJoin transaction.
- Compare the mean signed log₂W of a base tx against an augmented tx (adding inputs/outputs).
- Enumerate candidate change decompositions and rank them by mean signed log₂W.
- Validate the estimators against ground truth on real or synthetic data.

## Why this exists

A CoinJoin publishes a list of inputs and a list of outputs. An outside observer cannot see who sent what to whom, but they can **enumerate every plausible input→output mapping** the numbers allow. If only one mapping balances the books while still partitioning the inputs and outputs non-trivially, that is strong evidence of them being linked. If thousands do, an adversary will needs additional information in order to partition different users inputs and outputs correctly.

The core question is: *how many alternative mappings are there?* Counting them directly is exponential, so this tool uses `W(E)` — the number of input subsets summing to a given amount `E` — as the privacy primitive. A transaction with many subsets reaching its sub-sums has many possible decompositions, so many mappings, so good privacy.

Computing `W(E)` exactly is also exponential in the worst case, so the tool runs a **3-tier scale switch**:

1. **Exact enumeration / DP** for small sets — a true count.
2. **Block-convolution lookup** for medium sets — a proven lower bound.
3. **Asymptotic approximation** (Sasamoto / Toyoizumi / Nishimori) for large CoinJoins — fast, accurate at scale, but an approximation with error at finite N.

Commands that print all tiers side-by-side (`compare`, `full-report`) keep disagreements between estimators visible. Commands that return a single number (`measure`, `density`) pick the tier by regime and fall back to `min(Sasamoto, lookup)` once `N` is large enough for the asymptotic regime, so the reported number is never more optimistic than the proven lookup lower bound.

## Install

```bash
cargo build --release
export BIN=./target/release/dense-subset-sum

# Optional: used by the library's empirical-distribution tests (~353 MB).
bash scripts/fetch_cja_distribution.sh
```

## Run your first command in 60 seconds

The project ships with real Wasabi 2 transactions. Pick one and ask for the full report:

```bash
$BIN full-report --tx-label w2_6a6dcc22_17in6out
```

You'll see the transaction size and fee, the tx-level density `κ` (using `max(aᵢ)` as an ensemble range proxy), per-coin signed log₂W values, a radix check for standard denominations, and when the tx is small enough, a cross-check against the expensive Boltzmann/CJA ground truth. `κ_c` is a function of the target `E` (Sasamoto eq 4.3), not a single tx-level number, sweep `κ_c(x)` across the E-band with `$BIN dense-boundary`.

Then try the same on your own transaction:

```bash
$BIN full-report --inputs 100000,200000,300000 --outputs 150000,250000,200000
```

Or a JSON file (`{"label": "...", "inputs": [...], "outputs": [...]}`):

```bash
$BIN full-report --tx-json path/to/tx.json
```

## Other commands, by purpose

```bash
# Whole-tx mean signed log₂W and per-coin breakdown
$BIN measure       -i ... -o ...
$BIN coin-measures -i ... -o ...

# Count every plausible input→output mapping (small tx only)
$BIN analyze-tx    -i ... -o ...

# Mean signed log₂W: augmented tx vs base tx (reports before/after/delta)
$BIN compare-augmented -i ... -o ... --new-inputs 50000000 --new-outputs 50000000

# Rank change decompositions by mean signed log₂W
$BIN suggest-split -i ... -o ... --change 7168 --max-pieces 6

# Validate estimators against ground truth
$BIN compare-synthetic --n 50 --l-max 500
$BIN compare-wasabi2
$BIN compare-fixtures

# Full test suite
cargo test --release --lib
```

`-h` on any subcommand lists its flags.

## Glossary — the numbers you'll see

### Core quantities

- **N** — number of coins considered.
- **E** — a target amount in satoshis. The tool asks "how many subsets sum to E?" for diagnostic values of E (sub-transaction sums, Σ/2 as a midpoint, etc.).
- **W(E)** — the count of input subsets summing exactly to E. The privacy primitive: higher W = more alternative decompositions. `log₂ W` is reported because W gets astronomical.
- **Σa** — total input amount. E lives in `[0, Σa]`.

### Density parameters

- **κ = log₂(max value) / N** — how dense the inputs are. Low κ → subsets collide on many sums → ambiguity. High κ → unique sums → traceable. This is a single tx-level number; `max(aᵢ)` is used as a proxy for the ensemble range `L`.
- **κ_c(x)** — the critical density as a function of the normalized target `x = E / (N·L)` (paper eq. 4.3). The tx is in the **dense regime** at a given `E` when `κ < κ_c(x)`. Because κ_c depends on E, there is no single tx-level κ_c — use `dense-boundary` to sweep κ_c(x) across the E-band and find the dense range.
- **Regime tag**:
  - `EXACT` — small N: ground truth comes from enumeration. Ignore Sasamoto's error here.
  - `ASYMPTOTIC` — large N: the approximation is trustworthy.
  - `INTERMEDIATE` — cross-check before trusting a single estimator.
  - `Dense / Sparse / Unknown` — whether a proven lower bound confirmed density at this E.

### Estimators

Three ways of computing `log W(E)`, used together:

- **DP** — exact dynamic programming. Ground truth when the table fits in memory.
- **Lookup** — block-convolution, a proven lower bound.
- **Sasamoto** — asymptotic approximation. Fastest, accurate at large N, unreliable at small N or near the density boundary.

Accuracy columns:

- **`*_err`** — `(estimator − W_exact) / W_exact`, signed. Approximation error.
- **spearman** — rank correlation with ground truth (1.0 = perfect ordering). High Spearman with a big `*_err` means the approximation ranks things correctly but has a systematic scale bias.

### Per-coin measurements

- **signed multiset probe** — for each coin, "if I remove this coin, how many ways can the others be split (with signs) so the books still balance?"
- **log₂W_signed** — the log of that count. `N/A` = unreachable (no valid partition of the other coins balances the books without this one).

### Units

`log W` appears in **nats** (base *e*) in raw outputs and **log₂** (bits) in per-coin summaries. `W ≈ 2^(log₂ W)`: a coin with `log₂W_signed = 5.9` has roughly `2^5.9 ≈ 60` alternative arrangements.

### CJA mappings (ground truth for small txs)

- **#Map** — total input→output mappings consistent with the tx's balance.
- **#NonDer** — mappings not derivable from smaller ones (Maurer/Boltzmann).
- **log₂ M** — `log₂(#NonDer + 1)`, the privacy ground truth when the tx is small enough to enumerate.

### Radix analysis

For equal-amount CoinJoins (Wasabi 2, Whirlpool):

- **distinguished** — coins with low Hamming weight in base 2, 3, or 10 (round denominations like 50 M sats).
- **arbitrary** — the rest (change, leftovers).
- **denoms** — distinct standard denominations detected.

## Reading typical outputs

### `full-report`

```
κ = 1.5044, radix-like = true, estimator_picked = lookup
log W [lookup]: W = 0 (unreachable)
log W [Sasamoto]: log₂=-10.4138
estimator::estimate (log₂/N): 0.3401
```

`estimator_picked` shows which tier was used. `W = 0` at a single point doesn't contradict a globally dense tx — for the `κ_c(x)` profile across E and the dense E-band, run `dense-boundary`. `estimator::estimate` reports `log₂ W / N`, i.e. log₂W normalized per coin (comparable across transactions of different sizes).

### `coin-measures`

```
  in    0   50000000   5.907   0.6235
  in    5   49000000     N/A   0.6177
```

Columns:
- **role** — `in` (input) or `out` (output).
- **idx** — position in the input or output list.
- **value** — the coin's amount in satoshis.
- **log₂W_signed** — log₂ of the number of ± arrangements of the *other* coins that balance the books in this coin's place. `N/A` = no such arrangement exists (this coin is structurally required for the tx to balance).
- **κ_c** — critical density evaluated at this coin's natural `x = value / (N_in · max(inputs))`. Compare to the tx-level `κ`: coins with `κ < κ_c` are in the dense E-band at their own value. Per-coin view of κ/κ_c — no single "representative" E is chosen for the whole tx.

Coins of the same value share a measurement.

### `compare-wasabi2`

```
CJA mappings: 251 total, 150 non-derived
Unique sub-txs: 2
idx  in  out count  balance     lw_sas   lw_in   lw_out  lw_comb
  0   2    1   300  100000000    -9.68    1.10    1.39     2.48
```

Each row is a unique sub-transaction. `lw_sas / lw_in / lw_out / lw_comb` are `log W` from Sasamoto, lookup on inputs, lookup on outputs, and combined. `-inf` means no subset reaches that target on the lattice. SKIPPED = exceeds `--max-coins` (default 26).

### `compare-augmented`

```
Base tx: w2_6a6dcc22_17in6out (17 in / 6 out)
  + new inputs:  [50000000]
  + new outputs: [50000000]

Mean signed log₂W per coin:
  before: 4.9069
  after:  7.0340
  delta:  +2.1271
```

Reports how the mean signed log₂W per coin changes when the augmented coins are added.

- **before** — mean signed log₂W per coin on the *current* tx (no new coins).
- **after**  — same mean, recomputed on the *augmented* tx (current + `--new-inputs` + `--new-outputs`).
- **delta**  — `after − before`, in bits.

The reported number is the *mean* across all coins, so a positive delta can mask a specific UTXO that gained less than the others. Run `coin-measures` on the augmented tx to see the per-coin breakdown. Interpretation of the delta is left to the caller; this command is a measurement, not a decision rule.

### `dense-boundary` / `subset-density`

```
target-space dense range: E ∈ [2051082, 149729008] (fraction = 0.9902)
crossover k* (fraction_dense ≈ 0.5): 24.00
```

The band of E values where random subsets land in the dense regime, and the subset size `k*` at which a random subset is dense more than half the time.

## FAQ

**`full-report` says `W = 0 (unreachable)` but the regime is `Dense`. Contradiction?**
No. `κ < κ_c` describes the E *band*; `W(E) = 0` describes one E *point*. A sparse midpoint in an otherwise dense band is normal for irregular inputs. Use `dense-boundary` to inspect the band.

**Sasamoto returns `-inf` or a tiny number. Why?**
The asymptotic approximation requires the target to land on the GCD lattice of the inputs; when it doesn't, `-inf` is reported by design. When N is too small or the distribution is irregular, Sasamoto can also underestimate. Inspection commands print Sasamoto and lookup side-by-side so this stays visible, and single-number commands fall back to `min(Sasamoto, lookup)` at large N so a broken Sasamoto only makes that result stricter.

**Sasamoto's error is 99% but Spearman is 1.0. How?**
The approximation can have a systematic scale bias while still ranking E values correctly. Spearman captures the ranking (useful for comparing sub-transactions); absolute error captures the bias. The tool uses Sasamoto for the former, lookup/DP for the latter.

**`analyze-tx` reports `Entropy: 0.000 bits` on a tx with multiple mappings.**
Boltzmann entropy collapses to 0 when every mapping shares the same deterministic links. Check the `Non-derived mappings` list for the actual alternatives.

## Use as a library

```rust
use dense_subset_sum::{Transaction, estimator};

let tx = Transaction::new(vec![/* input sats */], vec![/* output sats */]);
let estimate = estimator::estimate(
    &tx.inputs,
    tx.inputs.iter().sum::<u64>() / 2,
    &Default::default(),
);
```

Any indexer output mapping to `Vec<u64>` of satoshis drops in. The scale switch picks the right tier automatically.

## Design notes

- **Proven vs approximated.** DP and lookup are proven lower bounds on W(E). Sasamoto is an asymptotic approximation with finite-N error. The tool labels every value with the tier that produced it and never reports an approximation as if it were a proof.
- **Tiered vs single-number.** Tiered commands print every tractable tier so disagreements are visible. Single-number commands fall back to `min(Sasamoto, lookup)` once `N` is in the asymptotic regime, so a broken approximation can only make that number stricter.
- **Fail-closed.** No sound positive lower bound ⇒ the regime is `Sparse` or `Unknown`, never a false `Dense`.
- **Radix composite.** For equal-denomination CoinJoins, a dedicated detector adds a conservative lower bound from denomination structure.

## References

1. Yuval Kogman ([nothingmuch](https://github.com/nothingmuch)) — privacy cost function specification.
2. Sasamoto, Toyoizumi, Nishimori — *Statistical Mechanics of Subset Sum* ([arxiv:cond-mat/0106125](https://arxiv.org/pdf/cond-mat/0106125)).
3. Maurer, Neudecker, Florian — *Anonymous CoinJoin Transactions with Arbitrary Values* (2017).
4. Maxwell — *CoinJoin: Bitcoin privacy for the real world* (bitcointalk, 2013).
5. LaurentMT — *Boltzmann / OXT entropy analysis* (Samourai, 2017).
6. Centre for Research on Cryptography and Security (CRoCS), Masaryk University — *coinjoin-analysis: processing and analysis of Wasabi/Whirlpool/JoinMarket coinjoin datasets*. GitHub repository, [crocs-muni/coinjoin-analysis](https://github.com/crocs-muni/coinjoin-analysis).