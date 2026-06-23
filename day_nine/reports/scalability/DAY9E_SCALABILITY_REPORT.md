# DAY9E Scalability Report

## Question

Does the Day 9d pure CMI learner survive scale?

## Method

The learner was not changed. Exact runs use `score(A,B) = min I(A;B|S)` over exhaustive conditioning sets up to `k`.
Large concept-space cases are marked `projected_only` when the exact conditioning-test count exceeds the run limit; these are not approximated recovery results.
Measured probe cost: `0.00024083` seconds per CMI test on this machine.

## Concept Count Scaling

| case | concepts | events | k | conditioning_tests | status | projected_runtime | memory_estimate_mb | f1 | reason |
|---|---|---|---|---|---|---|---|---|---|
| 60_concepts | 60 | 4000 | 3 | 57645360 | projected_only | 3.9h | 5.5756 | not_run | conditioning_tests_exceed_limit |
| 120_concepts | 120 | 4000 | 3 | 1955917320 | projected_only | 5.5d | 11.3159 | not_run | conditioning_tests_exceed_limit |
| 240_concepts | 240 | 4000 | 3 | 64446197040 | projected_only | 179.6d | 23.2910 | not_run | conditioning_tests_exceed_limit |
| 480_concepts | 480 | 4000 | 3 | 2092612051680 | projected_only | 16.0y | 49.2188 | not_run | conditioning_tests_exceed_limit |
| 960_concepts | 960 | 4000 | 3 | 67453966157760 | projected_only | 515.1y | 108.9844 | not_run | conditioning_tests_exceed_limit |

## Distractor Scaling

| case | distractors | concepts | events | k | conditioning_tests | status | projected_runtime | f1 | reason |
|---|---|---|---|---|---|---|---|---|---|
| +50 | 50 | 59 | 4000 | 3 | 52893854 | projected_only | 3.5h | not_run | conditioning_tests_exceed_limit |
| +100 | 100 | 109 | 4000 | 3 | 1202297904 | projected_only | 3.4d | not_run | conditioning_tests_exceed_limit |
| +250 | 250 | 259 | 4000 | 3 | 94530210054 | projected_only | 263.5d | not_run | conditioning_tests_exceed_limit |
| +500 | 500 | 509 | 4000 | 3 | 2808229480304 | projected_only | 21.4y | not_run | conditioning_tests_exceed_limit |
| +1000 | 1000 | 1009 | 4000 | 3 | 86548791458304 | projected_only | 660.9y | not_run | conditioning_tests_exceed_limit |

## Sample Complexity

| case | events | seeds | precision | recall | f1 | runtime_sec | rss_peak_mb |
|---|---|---|---|---|---|---|---|
| 100_events | 100 | 20 | 0.8780 | 0.9917 | 0.9267 | 0.3408 | 191.9375 |
| 500_events | 500 | 20 | 0.9714 | 1.0000 | 0.9846 | 0.3979 | 191.9375 |
| 1000_events | 1000 | 20 | 0.9714 | 1.0000 | 0.9846 | 0.4497 | 191.9375 |
| 4000_events | 4000 | 20 | 0.9929 | 1.0000 | 0.9962 | 0.6954 | 191.9375 |
| 10000_events | 10000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.1958 | 191.9375 |
| 50000_events | 50000 | 5 | 1.0000 | 1.0000 | 1.0000 | 4.6368 | 191.9375 |

Minimum tested event count with mean F1 >= 0.95: `500`.

## Conditioning Explosion

| case | confounders | concepts | k | conditioning_tests | precision | recall | f1 | runtime_sec | rss_peak_mb |
|---|---|---|---|---|---|---|---|---|---|
| 1_confounders | 1 | 5 | 1 | 40 | 1.0000 | 1.0000 | 1.0000 | 0.0052 | 191.9375 |
| 2_confounders | 2 | 6 | 2 | 165 | 1.0000 | 1.0000 | 1.0000 | 0.0380 | 191.9375 |
| 3_confounders | 3 | 7 | 3 | 546 | 1.0000 | 1.0000 | 1.0000 | 0.1503 | 191.9375 |
| 4_confounders | 4 | 8 | 4 | 1596 | 1.0000 | 1.0000 | 1.0000 | 0.6177 | 191.9375 |
| 5_confounders | 5 | 9 | 5 | 4320 | 1.0000 | 1.0000 | 1.0000 | 2.3154 | 191.9375 |
| 6_confounders | 6 | 10 | 6 | 11115 | 1.0000 | 1.0000 | 1.0000 | 8.3481 | 191.9375 |
| 7_confounders | 7 | 11 | 7 | 27610 | 1.0000 | 0.9600 | 0.9793 | 28.0557 | 191.9375 |
| 8_confounders | 8 | 12 | 8 | 66858 | 1.0000 | 0.9059 | 0.9479 | 88.9733 | 191.9375 |

## Event Sparsity

| case | concepts | events | k | conditioning_tests | precision | recall | f1 | runtime_sec | rss_peak_mb |
|---|---|---|---|---|---|---|---|---|---|
| distractor_rate_0.001 | 30 | 4000 | 1 | 12615 | 1.0000 | 1.0000 | 1.0000 | 1.7350 | 191.9375 |
| distractor_rate_0.005 | 30 | 4000 | 1 | 12615 | 1.0000 | 1.0000 | 1.0000 | 1.6846 | 191.9375 |
| distractor_rate_0.01 | 30 | 4000 | 1 | 12615 | 1.0000 | 1.0000 | 1.0000 | 1.8865 | 191.9375 |
| distractor_rate_0.03 | 30 | 4000 | 1 | 12615 | 1.0000 | 1.0000 | 1.0000 | 1.7162 | 191.9375 |
| distractor_rate_0.08 | 30 | 4000 | 1 | 12615 | 1.0000 | 1.0000 | 1.0000 | 1.7446 | 191.9375 |

## Answers

- Does recovery survive larger concept spaces? Not with exhaustive pure CMI at `k=3`. The requested 60+ concept runs exceed the exact-run budget before recovery quality is the limiting factor.
- What breaks first? Computation breaks first: the number of conditioning tests grows combinatorially with concept count and `k`.
- Statistical failure or computational failure? For the tested toy skeleton, sample complexity is manageable at small concept count; large-space failure is computational before it is statistical.
- Estimated maximum feasible concept count on current hardware: about `9` concepts for exact full-matrix `k=3` under this script's conservative interactive limit; 60 concepts already projects beyond practical interactive runtime.
- Is CMI worth continuing? Yes as a diagnostic/statistical direction, but not as a naive all-pairs exhaustive learner.
- Next bottleneck after confound rejection: candidate-set control. The project needs a principled way to restrict which pairs and conditioning variables are tested without reintroducing the Day 9b heuristic failure.

## Verdict

Day 9e changes the interpretation of Day 9d: pure CMI works as a clean causal-skeleton statistic, but exhaustive graph recovery does not scale to realistic concept spaces.
The next research problem is not another learner gate; it is scalable candidate generation that preserves the valid conditioning sets Day 9c/9d proved are necessary.

The 60-concept requested case was intentionally not approximated as a pass/fail recovery run; the report records it as a computational scaling failure.