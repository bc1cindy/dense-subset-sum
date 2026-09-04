# dense-subset-sum

> **Work in progress.**

A tool to measure the **ambiguity of a CoinJoin's amount channel** using conservatively counted subset-sum explanations, from an adversary's point of view.

Given the inputs and outputs of a Bitcoin CoinJoin, it counts subset-sum solutions that can support alternative explanations of the amounts. The count itself is the output; interpretation of what counts as "enough" is left to the caller.

**Scope of the number.** `W` is an *attacker's ambiguity count*, not a privacy score. It measures one channel (the amounts) of one transaction, under **no auxiliary information**. Exact methods report exact counts; truncated sparse methods report explicit lower bounds; the Sasamoto path reports an asymptotic estimate with no one-sided guarantee. A **low exact count** is informative linkage evidence. A low lower bound is inconclusive because the true count may be much higher; a high lower bound proves only that many amount explanations exist under the selected model. An approximation is diagnostic and cannot certify either conclusion. None of these results establishes transaction privacy, which remains a whole-graph question conditioned on the adversary's auxiliary information. Downstream consumers must preserve the result kind; mapping probabilities come from the separate diagnostic mapping API described below.

## Why this exists

A CoinJoin publishes a list of inputs and a list of outputs. An outside observer cannot see who sent what to whom, but they can **enumerate every plausible input→output mapping** the numbers allow. If only one mapping balances the books while still partitioning the inputs and outputs non-trivially, that is strong evidence of them being linked. If thousands do, an adversary will need additional information in order to partition different users inputs and outputs correctly.

The production-oriented question is: *how much subset-sum cover do the other participants' amounts provide?* The primitive `W(E)` counts input subsets summing to a target `E`. It is related to, but is not the same mathematical object as, a count of complete sub-transaction mappings.

Computing `W(E)` exactly is also exponential in the worst case, so the tool exposes **four counting primitives**, picked by the caller:

1. **Brute force / DP** — small N (exponential enumeration is tractable).
2. **Radix** — independent of N, exploits output structure; counts `Σ k × m!` mappings.
3. **Sparse convolution** — medium N (scales until the sumset table blows up).
4. **Asymptotic approximation** (Sasamoto / Toyoizumi / Nishimori) — large N (asymptotic, not valid for small W).

## Diagnostic mapping API

The optional Python extension exposes `mapping_analysis(inputs, outputs, budget_ms=None)` and
`pairwise_link_prob(...)` for differential research against exact sub-transaction oracles. This API
reports the mapping family selected by the bundled CJA-based implementation, its entropy, and links
on which that restricted family agrees. It is diagnostic evidence, not `W(E)`, not CoinScore, and
not a certificate of transaction or whole-graph privacy.
When the repeated-denomination fast path skips enumeration, `mapping_analysis` reports
`dense_fast_path` and `pairwise_link_prob` returns `None`; no probability matrix is inferred.

Programs that write publishable results must call `dss.require_build_revision()`. Such builds must
set `DSS_GIT_REV` to the immutable 40-character Git revision used to compile the extension. Local
interactive builds may leave it unset; `dss.__rev__` will then be `None` and the publication gate
will fail explicitly.

Dependency maintenance and the current `nom` migration blocker are recorded in
[`docs/dependency-maintenance.md`](docs/dependency-maintenance.md).

## Glossary — the numbers you'll see

### Core quantities

- **N** — number of coins considered.
- **E** — a target amount in satoshis. The tool asks "how many subsets sum to E?" for diagnostic values of E (sub-transaction sums, Σ/2 as a midpoint, etc.).
- **W(E)** — the count of input subsets summing exactly to E. The ambiguity primitive: higher W = more alternative decompositions (amount channel silent); a *low* W is the informative case — it links coins. `log₂ W` is reported because W gets astronomical.
- **Σa** — total input amount. E lives in `[0, Σa]`.

### Density parameters

- **κ = log₂(max value) / N** — how dense the inputs are. Low κ → subsets collide on many sums → ambiguity. High κ → unique sums → traceable. This is a single tx-level number; `max(aᵢ)` is used as a proxy for the ensemble range `L`.
- **κ_c(x)** — the critical density as a function of the normalized target `x = E / (N·L)` (paper eq. 4.3). The tx is in the **dense regime** at a given `E` when `κ < κ_c(x)`. Because κ_c depends on E, the dense regime must be evaluated per target E rather than once per transaction.

## References

1. Yuval Kogman ([nothingmuch](https://github.com/nothingmuch))
2. Sasamoto, Toyoizumi, Nishimori — *Statistical Mechanics of Subset Sum* ([arxiv:cond-mat/0106125](https://arxiv.org/pdf/cond-mat/0106125)).
3. Maurer, Neudecker, Florian — *Anonymous CoinJoin Transactions with Arbitrary Values* (2017).
4. Maxwell — *CoinJoin: Bitcoin privacy for the real world* (bitcointalk, 2013).
5. LaurentMT — *Boltzmann / OXT entropy analysis* (Samourai, 2017).
6. Centre for Research on Cryptography and Security (CRoCS), Masaryk University — *coinjoin-analysis: processing and analysis of Wasabi/Whirlpool/JoinMarket coinjoin datasets*. GitHub repository, [crocs-muni/coinjoin-analysis](https://github.com/crocs-muni/coinjoin-analysis).
7. Bringmann, Fischer, Nakos — *Deterministic and Las Vegas Algorithms for Sparse Nonnegative Convolution* ([arXiv:2107.07625](https://arxiv.org/abs/2107.07625)).
