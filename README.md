# dense-subset-sum

> **Work in progress.**

A tool to measure **how private a CoinJoin transaction is**.

Given the inputs and outputs of a Bitcoin CoinJoin, it counts (or approximates) how many alternative ways an observer could plausibly match inputs to outputs. The count itself is the output; interpretation of what counts as "enough" is left to the caller.

## Why this exists

A CoinJoin publishes a list of inputs and a list of outputs. An outside observer cannot see who sent what to whom, but they can **enumerate every plausible input→output mapping** the numbers allow. If only one mapping balances the books while still partitioning the inputs and outputs non-trivially, that is strong evidence of them being linked. If thousands do, an adversary will needs additional information in order to partition different users inputs and outputs correctly.

The core question is: *how many alternative mappings are there?* Counting them directly is exponential, so this tool uses `W(E)` — the number of input subsets summing to a given amount `E` — as the privacy primitive. A transaction with many subsets reaching its sub-sums has many possible decompositions, so many mappings, so good privacy.

Computing `W(E)` exactly is also exponential in the worst case, so the tool runs a **3-tier scale switch**:

1. **Exact enumeration / DP** for small sets — a true count.
2. **Block-convolution lookup** for medium sets — a proven lower bound.
3. **Asymptotic approximation** (Sasamoto / Toyoizumi / Nishimori) for large CoinJoins — fast, accurate at scale, but an approximation with error at finite N.

## Glossary — the numbers you'll see

### Core quantities

- **N** — number of coins considered.
- **E** — a target amount in satoshis. The tool asks "how many subsets sum to E?" for diagnostic values of E (sub-transaction sums, Σ/2 as a midpoint, etc.).
- **W(E)** — the count of input subsets summing exactly to E. The privacy primitive: higher W = more alternative decompositions. `log₂ W` is reported because W gets astronomical.
- **Σa** — total input amount. E lives in `[0, Σa]`.

### Density parameters

- **κ = log₂(max value) / N** — how dense the inputs are. Low κ → subsets collide on many sums → ambiguity. High κ → unique sums → traceable. This is a single tx-level number; `max(aᵢ)` is used as a proxy for the ensemble range `L`.
- **κ_c(x)** — the critical density as a function of the normalized target `x = E / (N·L)` (paper eq. 4.3). The tx is in the **dense regime** at a given `E` when `κ < κ_c(x)`. Because κ_c depends on E, there is no single tx-level κ_c — use `dense-boundary` to sweep κ_c(x) across the E-band and find the dense range.

### Estimators

Three ways of computing `log W(E)`, used together:

- **DP** — exact dynamic programming. Ground truth when the table fits in memory.
- **Lookup** — block-convolution, a proven lower bound.
- **Sasamoto** — asymptotic approximation. Fastest, accurate at large N, unreliable at small N or near the density boundary.

## References

1. Yuval Kogman ([nothingmuch](https://github.com/nothingmuch)) — [*A mechanism for improving CoinJoin anonymity sets and sybil resistance*](https://gist.github.com/nothingmuch/f5b9a559958c6116606d9da0d4d884f2).
2. Sasamoto, Toyoizumi, Nishimori — *Statistical Mechanics of Subset Sum* ([arxiv:cond-mat/0106125](https://arxiv.org/pdf/cond-mat/0106125)).
3. Maurer, Neudecker, Florian — *Anonymous CoinJoin Transactions with Arbitrary Values* (2017).
4. Maxwell — *CoinJoin: Bitcoin privacy for the real world* (bitcointalk, 2013).
5. LaurentMT — *Boltzmann / OXT entropy analysis* (Samourai, 2017).
6. Centre for Research on Cryptography and Security (CRoCS), Masaryk University — *coinjoin-analysis: processing and analysis of Wasabi/Whirlpool/JoinMarket coinjoin datasets*. GitHub repository, [crocs-muni/coinjoin-analysis](https://github.com/crocs-muni/coinjoin-analysis).