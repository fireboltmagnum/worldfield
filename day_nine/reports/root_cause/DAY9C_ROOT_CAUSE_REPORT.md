# DAY9C Root Cause Report

## Core Question

Does the causal signal already exist in the statistics and the current learner throw it away,
or is the current PMI+screening family fundamentally incapable of recovering multi-confound causal structure?

## Short Answer

For **double-confound** and **triple-confound** worlds, the causal signal already exists in the statistics.
The current Day 9b learner throws that signal away before it can be used.

More precisely:
- the correct joint conditioning sets are generated
- their conditional information is near zero
- the current weakest-link gate rejects them before scoring
- a pure exhaustive conditional-information baseline up to set size 3 cleanly separates confounded from direct edges for 1-, 2-, and 3-confound worlds
- the limitation only becomes fundamental when the number of confounders exceeds the conditioning budget (4 and 5 confounders with max conditioning set size 3)

So the rigorous answer is:

> Day 9b fails on 2- and 3-confound worlds primarily because of implementation/heuristic limitations, not because the signal is absent or because conditional-information methods are fundamentally too weak at that scale.

## Failed-Run Diagnostics

### double_confound
- failed runs: 50/50
- output file: `double_confound_failed_diagnostics.jsonl`
- `PMI(A,B)`: 0.706076 +/- 0.020136
- `I(A;B)`: 0.152967 +/- 0.006263
- `I(A;B|C)`: 0.087129 +/- 0.006064
- `I(A;B|D)`: 0.086529 +/- 0.006306
- `I(A;B|C,D)`: 0.000463 +/- 0.000314
- correct/best explanatory set skipped by current gate: 50/50
- no subset survived the current gate for actual testing: 50/50

### triple_confound
- failed runs: 50/50
- output file: `triple_confound_failed_diagnostics.jsonl`
- `PMI(A,B)`: 0.689096 +/- 0.015719
- `I(A;B)`: 0.162708 +/- 0.008309
- `I(A;B|C)`: 0.122247 +/- 0.007667
- `I(A;B|D)`: 0.118545 +/- 0.008242
- `I(A;B|C,D)`: 0.070817 +/- 0.006804
- `I(A;B|E)`: 0.117842 +/- 0.006835
- `I(A;B|C,D,E)`: 0.005742 +/- 0.001703
- correct/best explanatory set skipped by current gate: 50/50
- no subset survived the current gate for actual testing: 50/50

## Learner Decision Path

For every failed run, the emitted diagnostics include:
- candidate conditioning sets examined
- conditional-information score for each set
- weakest-link value used by the current gate
- whether the set was skipped before scoring could affect the decision
- best tested set
- best oracle set

In both double- and triple-confound files, the best oracle set is the correct joint explanatory set and it is always skipped by the current gate.

## Pure Conditional-Information Baseline (No Gates, No Heuristics)

Baseline definition:
- For a pair `(X,Y)`, compute the exhaustive minimum of `I(X;Y|S)` over all conditioning sets `S` of size `0..3`.
- No weakest-link filters.
- No PMI-weight gates.
- No triangle heuristics.

This is diagnostic only. It is not a new learner.

| Structure | mean min-CMI(confounded edge) | max min-CMI(confounded edge) | mean min-CMI(direct edges) | min min-CMI(direct edges) | clean margin? |
|---|---:|---:|---:|---:|---:|
| confound | 0.000268 | 0.001261 | 0.142536 | 0.122273 | YES |
| double_confound | 0.000517 | 0.002499 | 0.061680 | 0.045288 | YES |
| triple_confound | 0.005770 | 0.009675 | 0.035249 | 0.018291 | YES |

Interpretation:
- For single-, double-, and triple-confound worlds, a pure exhaustive conditional-information baseline already has a clean statistical margin between the confounded `A-B` edge and the true direct edges.
- Therefore the current Day 9b failure on 2 and 3 confounders is not because the statistics are insufficient.
- It is because the current heuristic prevents the model from using the available signal.

## Scaling Curves: 1-5 Confounders

The next question is where the method family itself starts breaking as the number of confounders increases.

| # confounders | mean min-CMI(A-B), sets<=3 | max min-CMI(A-B), sets<=3 | mean min-CMI(direct edges), sets<=3 | min min-CMI(direct edges), sets<=3 | mean oracle I(A;B|all confounds) | clean margin? |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.000286 | 0.001295 | 0.144679 | 0.122865 | 0.000287 | YES |
| 2 | 0.000535 | 0.001483 | 0.061764 | 0.046581 | 0.000535 | YES |
| 3 | 0.000982 | 0.002549 | 0.039494 | 0.026348 | 0.000982 | YES |
| 4 | 0.049764 | 0.060234 | 0.030249 | 0.019862 | 0.001428 | NO |
| 5 | 0.074503 | 0.085548 | 0.023444 | 0.012844 | 0.002132 | NO |

Interpretation:
- For 1, 2, and 3 confounders, exhaustive conditioning up to set size 3 is enough. The confounded edge remains well below the direct-edge distribution.
- For 4 and 5 confounders, the oracle full-set conditional information is still near zero, so the signal still exists in the statistics.
- But exhaustive conditioning restricted to sets of size <=3 can no longer reach the needed full explanatory set, and the confounded-edge distribution rises into the direct-edge range.
- That is the first point where the limitation becomes fundamental to the chosen conditioning-budget family rather than a mere implementation bug.

## Root Cause

### What fails in the current implementation
1. The current set-conditioned learner generates the correct joint sets.
2. It computes the right kind of statistic in principle.
3. But the pre-gate `base_ij >= 0.95 * weakest_link` rejects the correct joint set before the conditional-information score can decide the edge.
4. So the multi-confound failure in Day 9b is immediate and deterministic for the current benchmark worlds.

### What fails in the naive family
If the gate is removed naively and the same suppression rule is left in place, the family over-suppresses direct edges and breaks single-confound, chain, and collider cases.

That means:
- the failure is **not only** an implementation bug
- but it is also **not** evidence that conditional-information methods are too weak for 2-3 confounders
- instead, the current PMI + screening heuristic is the wrong mechanism for harvesting a signal that the statistics already contain

## Final Answer To The Core Question

For the current Day 9 benchmarks:
- **2 and 3 confounders:** the signal already exists in the statistics and the current learner throws it away
- **4 and 5 confounders with max conditioning size 3:** the signal still exists in the oracle full conditioning, but the restricted family cannot recover it without enlarging the conditioning set budget

So the correct root-cause statement is:

> Day 9b does not fail because multi-confound structure is absent from the statistics. It fails because the current heuristic screening rule blocks the correct joint explanations, and because the restricted PMI+screening implementation is not a safe way to exploit higher-order conditional information. The broader conditional-information family is sufficient through 3 confounders under exhaustive conditioning, and becomes fundamentally limited only once the number of confounders exceeds the conditioning-set budget.

## Deliverables

- failed run diagnostics: `double_confound_failed_diagnostics.jsonl`, `triple_confound_failed_diagnostics.jsonl`
- scaling csv: `conditional_information_scaling.csv`
- this report: `DAY9C_ROOT_CAUSE_REPORT.md`