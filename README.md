# dense-subset-sum

A tool to measure **how private a CoinJoin transaction is**.

Given the inputs and outputs of a Bitcoin CoinJoin, it counts (or approximates) how many alternative ways an observer could plausibly match inputs to outputs. More alternatives = more privacy. Fewer = the transaction leaks.

You can use it to:

- Score an existing CoinJoin transaction.
- Decide whether joining a co-spend proposal actually helps.
- Suggest how to split a change output to maximise ambiguity.
- Validate the estimators against ground truth on real or synthetic data.

## Why this exists

A CoinJoin publishes a list of inputs and a list of outputs. An outside observer cannot see who sent what to whom, but they can **enumerate every plausible input→output mapping** the numbers allow. If only one mapping balances the books, the transaction is traceable. If thousands do, the observer can't tell them apart — that's privacy.

The core question is: *how many alternative mappings are there?* Counting them directly is exponential, so this tool uses `W(E)` — the number of input subsets summing to a given amount `E` — as the privacy primitive. A transaction with many subsets reaching its sub-sums has many possible decompositions, so many mappings, so good privacy.

Computing `W(E)` exactly is also exponential in the worst case, so the tool runs a **3-tier scale switch**:

1. **Exact enumeration / DP** for small sets — a true count.
2. **Block-convolution lookup** for medium sets — a proven lower bound.
3. **Asymptotic approximation** (Sasamoto / Toyoizumi / Nishimori) for large CoinJoins — fast, accurate at scale, but an approximation with error at finite N.

The tool always reports the *more conservative* of the applicable tiers. Privacy is never overstated.

## Install

```bash
cargo build --release
export BIN=./target/release/dense-subset-sum

# Optional, for the real Bitcoin UTXO distribution (~353 MB):
bash scripts/fetch_cja_distribution.sh
```

## Run your first command in 60 seconds

The project ships with real Wasabi 2 transactions. Pick one and ask for the full report:

```bash
$BIN full-report --tx-label w2_6a6dcc22_17in6out
```

You'll see the transaction size and fee, density parameters (`κ`, `κ_c`), a per-coin privacy score, a radix check for standard denominations, and — when the tx is small enough — a cross-check against the expensive Boltzmann/CJA ground truth.

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
# Whole-tx privacy score and per-coin breakdown
$BIN cost         -i ... -o ...
$BIN coin-scores  -i ... -o ...

# Count every plausible input→output mapping (small tx only)
$BIN analyze-tx   -i ... -o ...

# Should I add my UTXO to this co-spend? (before/after/delta)
$BIN marginal-score -i ... -o ... --new-inputs 50000000 --new-outputs 50000000

# Decompose a change amount to maximise ambiguity
$BIN suggest-split -i ... -o ... --change 7168 --max-pieces 6

# Validate estimators against ground truth
$BIN compare-synthetic --n 50 --l-max 500
$BIN compare-wasabi2
$BIN compare-empirical --n 50 --samples 2000000 --divisor 1000000
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

- **κ = log₂(max value) / N** — how dense the inputs are. Low κ → subsets collide on many sums → ambiguity. High κ → unique sums → traceable.
- **κ_c** — the critical density for the given E (paper eq. 4.3). The transaction is in the **dense regime** when `κ < κ_c`. Dense means the approximation is meaningful; sparse means W is likely small or zero.
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

### Per-coin scores

- **signed multiset probe** — for each coin, "if I remove this coin, how many ways can the others be split (with signs) so the books still balance?"
- **log₂W_signed** — the log of that count. Higher = more interchangeable. `N/A` = unreachable, the coin has a unique role and leaks.

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
density_regime at Σin/2: κ = 1.5044, κ_c = 0.9817
log W [lookup]: W = 0 (unreachable)
log W [Sasamoto]: log₂=-10.4138
loss::score (log₂/N): 0.3401
```

`estimator_picked` shows which tier was used. `κ > κ_c` at this E means sparse — `W = 0` at a single point doesn't contradict a globally dense tx. `loss::score` is the headline `log₂ W / N`, comparable across transactions.

### `coin-scores`

```
  in    0   50000000   5.907
  in    5   49000000   N/A
```

Coins of the same value share a score (they are interchangeable). `N/A` = leak.

### `compare-wasabi2`

```
CJA mappings: 251 total, 150 non-derived
Unique sub-txs: 2
idx  in  out count  balance     lw_sas   lw_in   lw_out  lw_comb
  0   2    1   300  100000000    -9.68    1.10    1.39     2.48
```

Each row is a unique sub-transaction. `lw_sas / lw_in / lw_out / lw_comb` are `log W` from Sasamoto, lookup on inputs, lookup on outputs, and combined. `-inf` means no subset reaches that target on the lattice. SKIPPED = exceeds `--max-coins` (default 26).

### `marginal-score`

```
Base tx: w2_6a6dcc22_17in6out (17 in / 6 out)
  + new inputs:  [50000000]
  + new outputs: [50000000]

Mean signed log₂W per coin:
  before: 4.9069
  after:  7.0340
  delta:  +2.1271

→ JOIN: adding these coins increases privacy.
```

Answers "is it worth adding my UTXO to this co-spend?"

- **before** — mean signed log₂W per coin on the *current* tx (no new coins). Baseline privacy.
- **after**  — same mean, recomputed on the *augmented* tx (current + `--new-inputs` + `--new-outputs`).
- **delta**  — `after − before`, in bits. Positive ⇒ adding these coins lifts the average ambiguity; negative ⇒ your UTXO sticks out and drags the average down.

`delta > 0` ⇒ JOIN. The score is the *mean* across all coins. A positive delta can hide that your specific UTXO gained less than the others. Run `coin-scores` on the augmented tx to see your row.

### `dense-boundary` / `subset-density`

```
target-space dense range: E ∈ [2051082, 149729008] (fraction = 0.9902)
crossover k* (fraction_dense ≈ 0.5): 24.00
```

The band of E values where random subsets land in the dense regime, and the subset size `k*` at which a random subset is dense more than half the time.

### `compare-empirical`

```
Rescaled by /1000: kept 27/30 non-zero values
N=27 > 25: switching to Monte Carlo (500000 samples, 60000 ms cap)
```

`--divisor` rescales sat-level values. At raw satoshi granularity every subset sum is unique (`W range: [0, 0]`), so work at mBTC (`--divisor 1000`) or BTC-scale (`--divisor 1000000`). At `N > 25`, Monte Carlo takes over.

## FAQ

**`full-report` says `W = 0 (unreachable)` but the regime is `Dense`. Contradiction?**
No. `κ < κ_c` describes the E *band*; `W(E) = 0` describes one E *point*. A sparse midpoint in an otherwise dense band is normal for irregular inputs. Use `dense-boundary` to inspect the band.

**Sasamoto returns `-inf` or a tiny number. Why?**
The asymptotic approximation requires the target to land on the GCD lattice of the inputs; when it doesn't, `-inf` is reported by design. When N is too small or the distribution is irregular, Sasamoto can also underestimate — the scale switch takes `min(Sasamoto, lookup)`, so a broken Sasamoto only makes the result more conservative.

**Sasamoto's error is 99% but Spearman is 1.0. How?**
The approximation can have a systematic scale bias while still ranking E values correctly. Spearman captures the ranking (useful for comparing sub-transactions); absolute error captures the bias. The tool uses Sasamoto for the former, lookup/DP for the latter.

**`compare-empirical --divisor 1` shows `W range: [0, 0]`. Why is the divisor needed?**
Real Bitcoin UTXOs hold arbitrary 64-bit sat values (e.g. 47 382 719, 128 394 021). No two distinct subsets of a few dozen such values ever land on the same sum by chance, so every `W(E) ∈ {0, 1}` — there is no density to measure. The divisor floor-divides each value into a coarser grid (mBTC with `/1000`, BTC-scale with `/1000000`), so nearby values fall into the same bucket and subsets start colliding. `compare-wasabi2` doesn't need a divisor because Wasabi 2 uses round denominations (50 M, 100 M sat…) that collide naturally.

**`analyze-tx` reports `Entropy: 0.000 bits` on a tx with multiple mappings.**
Boltzmann entropy collapses to 0 when every mapping shares the same deterministic links. Check the `Non-derived mappings` list for the actual alternatives.

## Use as a library

```rust
use dense_subset_sum::{Transaction, loss};

let tx = Transaction::new(vec![/* input sats */], vec![/* output sats */]);
let score = loss::score(
    &tx.inputs,
    tx.inputs.iter().sum::<u64>() / 2,
    &Default::default(),
);
```

Any indexer output mapping to `Vec<u64>` of satoshis drops in. The scale switch picks the right tier automatically.

## Design notes

- **Proven vs approximated.** DP and lookup are proven lower bounds on W(E). Sasamoto is an asymptotic approximation with finite-N error. The tool tells you which tier it used and never reports an approximation as if it were a proof.
- **Conservative combination.** When both Sasamoto and lookup apply, the tool takes the minimum — a broken approximation can only make the estimate stricter.
- **Fail-closed.** No sound positive lower bound ⇒ the regime is `Sparse` or `Unknown`, never a false `Dense`.
- **Radix composite.** For equal-denomination CoinJoins, a dedicated detector adds a conservative lower bound from denomination structure.

## References

1. Yuval Kogman ([nothingmuch](https://github.com/nothingmuch)) — privacy cost function specification.
2. Sasamoto, Toyoizumi, Nishimori — *Statistical Mechanics of Subset Sum* ([arxiv:cond-mat/0106125](https://arxiv.org/pdf/cond-mat/0106125)).
3. Maurer, Neudecker, Florian — *Anonymous CoinJoin Transactions with Arbitrary Values* (2017).
4. Maxwell — *CoinJoin: Bitcoin privacy for the real world* (bitcointalk, 2013).
5. LaurentMT — *Boltzmann / OXT entropy analysis* (Samourai, 2017).
