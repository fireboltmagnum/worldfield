# DAY9D Causal Graph Report

## Question

Can a graph built from experience recover concept-level causal skeleton structure, not fragment-scale relations, or only reject simple confounds?

## Short Answer

On these concept-level benchmark worlds, pure conditional-information graph recovery can recover the undirected causal skeleton substantially better than PMI and the Day 9/9b heuristic screeners.
It is still not causal direction discovery, and it still depends on the conditioning-set budget matching the number of jointly explanatory confounders.

## Learner Summary

| learner | precision | recall | f1 |
|---|---|---|---|
| pmi | 0.398 | 1.000 | 0.557 |
| day9_single_screen | 0.639 | 0.962 | 0.761 |
| day9b_set_screen | 0.639 | 0.962 | 0.761 |
| pure_cmi_k1 | 0.917 | 1.000 | 0.954 |
| pure_cmi_k2 | 0.945 | 1.000 | 0.970 |
| pure_cmi_k3 | 0.966 | 1.000 | 0.981 |
| pc_style_k3 | 0.966 | 1.000 | 0.981 |
| partial_corr | 0.766 | 1.000 | 0.863 |
| sparse_regression | 0.752 | 1.000 | 0.853 |

## Edge Recovery By World

| world | learner | precision | recall | f1 | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| single_confound | pmi | 0.362 | 1.000 | 0.519 | 3 | 6.240 | 0 |
| single_confound | day9_single_screen | 0.621 | 1.000 | 0.762 | 3 | 1.940 | 0 |
| single_confound | day9b_set_screen | 0.621 | 1.000 | 0.762 | 3 | 1.940 | 0 |
| single_confound | pure_cmi_k1 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| single_confound | pure_cmi_k2 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| single_confound | pure_cmi_k3 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| single_confound | pc_style_k3 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| single_confound | partial_corr | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| single_confound | sparse_regression | 0.990 | 1.000 | 0.994 | 3 | 0.040 | 0 |
| double_confound | pmi | 0.413 | 1.000 | 0.576 | 5 | 7.930 | 0 |
| double_confound | day9_single_screen | 0.582 | 0.912 | 0.707 | 4.560 | 3.350 | 0.440 |
| double_confound | day9b_set_screen | 0.582 | 0.912 | 0.707 | 4.560 | 3.350 | 0.440 |
| double_confound | pure_cmi_k1 | 0.833 | 1.000 | 0.909 | 5 | 1 | 0 |
| double_confound | pure_cmi_k2 | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 |
| double_confound | pure_cmi_k3 | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 |
| double_confound | pc_style_k3 | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 |
| double_confound | partial_corr | 0.720 | 1.000 | 0.837 | 5 | 1.950 | 0 |
| double_confound | sparse_regression | 0.707 | 1.000 | 0.827 | 5 | 2.120 | 0 |
| triple_confound | pmi | 0.508 | 1.000 | 0.667 | 7 | 7.360 | 0 |
| triple_confound | day9_single_screen | 0.716 | 0.861 | 0.778 | 6.030 | 2.410 | 0.970 |
| triple_confound | day9b_set_screen | 0.716 | 0.861 | 0.778 | 6.030 | 2.410 | 0.970 |
| triple_confound | pure_cmi_k1 | 0.875 | 1.000 | 0.933 | 7 | 1 | 0 |
| triple_confound | pure_cmi_k2 | 0.875 | 1.000 | 0.933 | 7 | 1 | 0 |
| triple_confound | pure_cmi_k3 | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| triple_confound | pc_style_k3 | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| triple_confound | partial_corr | 0.634 | 1.000 | 0.776 | 7 | 4.040 | 0 |
| triple_confound | sparse_regression | 0.623 | 1.000 | 0.767 | 7 | 4.270 | 0 |
| chain | pmi | 0.386 | 1.000 | 0.543 | 3 | 5.750 | 0 |
| chain | day9_single_screen | 0.649 | 1.000 | 0.779 | 3 | 1.810 | 0 |
| chain | day9b_set_screen | 0.649 | 1.000 | 0.779 | 3 | 1.810 | 0 |
| chain | pure_cmi_k1 | 0.805 | 1.000 | 0.889 | 3 | 0.780 | 0 |
| chain | pure_cmi_k2 | 0.805 | 1.000 | 0.889 | 3 | 0.780 | 0 |
| chain | pure_cmi_k3 | 0.807 | 1.000 | 0.890 | 3 | 0.770 | 0 |
| chain | pc_style_k3 | 0.807 | 1.000 | 0.890 | 3 | 0.770 | 0 |
| chain | partial_corr | 0.745 | 1.000 | 0.853 | 3 | 1.050 | 0 |
| chain | sparse_regression | 0.732 | 1.000 | 0.844 | 3 | 1.140 | 0 |
| collider | pmi | 0.411 | 1.000 | 0.568 | 3 | 5.180 | 0 |
| collider | day9_single_screen | 0.608 | 1.000 | 0.751 | 3 | 2.090 | 0 |
| collider | day9b_set_screen | 0.608 | 1.000 | 0.751 | 3 | 2.090 | 0 |
| collider | pure_cmi_k1 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| collider | pure_cmi_k2 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| collider | pure_cmi_k3 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| collider | pc_style_k3 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| collider | partial_corr | 0.750 | 1.000 | 0.857 | 3 | 1 | 0 |
| collider | sparse_regression | 0.745 | 1.000 | 0.853 | 3 | 1.040 | 0 |
| mixed | pmi | 0.312 | 1.000 | 0.469 | 6 | 14.580 | 0 |
| mixed | day9_single_screen | 0.656 | 1.000 | 0.790 | 6 | 3.240 | 0 |
| mixed | day9b_set_screen | 0.656 | 1.000 | 0.790 | 6 | 3.240 | 0 |
| mixed | pure_cmi_k1 | 0.991 | 1.000 | 0.995 | 6 | 0.060 | 0 |
| mixed | pure_cmi_k2 | 0.991 | 1.000 | 0.995 | 6 | 0.060 | 0 |
| mixed | pure_cmi_k3 | 0.991 | 1.000 | 0.995 | 6 | 0.060 | 0 |
| mixed | pc_style_k3 | 0.991 | 1.000 | 0.995 | 6 | 0.060 | 0 |
| mixed | partial_corr | 0.744 | 1.000 | 0.853 | 6 | 2.070 | 0 |
| mixed | sparse_regression | 0.717 | 1.000 | 0.834 | 6 | 2.430 | 0 |

## Best Learner Per World

| world | learner | precision | recall | f1 |
|---|---|---|---|---|
| chain | pure_cmi_k3 | 0.807 | 1.000 | 0.890 |
| collider | pure_cmi_k1 | 1.000 | 1.000 | 1.000 |
| double_confound | pure_cmi_k2 | 1.000 | 1.000 | 1.000 |
| mixed | pure_cmi_k1 | 0.991 | 1.000 | 0.995 |
| single_confound | pure_cmi_k1 | 1.000 | 1.000 | 1.000 |
| triple_confound | pure_cmi_k3 | 1.000 | 1.000 | 1.000 |

## Conditioning-Set Scaling

| n_confounds | conditioning_k | precision | recall | f1 | mean_margin | min_margin | best_separation |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.000 | 1.000 | 1.000 | 0.138 | 0.114 | True |
| 1 | 2 | 1.000 | 1.000 | 1.000 | 0.138 | 0.114 | True |
| 1 | 3 | 1.000 | 1.000 | 1.000 | 0.138 | 0.114 | True |
| 1 | 4 | 1.000 | 1.000 | 1.000 | 0.138 | 0.114 | True |
| 1 | 5 | 1.000 | 1.000 | 1.000 | 0.138 | 0.114 | True |
| 2 | 1 | 0.833 | 1.000 | 0.909 | -0.030 | -0.044 | False |
| 2 | 2 | 1.000 | 1.000 | 1.000 | 0.054 | 0.047 | True |
| 2 | 3 | 1.000 | 1.000 | 1.000 | 0.054 | 0.047 | True |
| 2 | 4 | 1.000 | 1.000 | 1.000 | 0.054 | 0.047 | True |
| 2 | 5 | 1.000 | 1.000 | 1.000 | 0.054 | 0.047 | True |
| 3 | 1 | 0.875 | 1.000 | 0.933 | -0.081 | -0.098 | False |
| 3 | 2 | 0.875 | 1.000 | 0.933 | -0.028 | -0.042 | False |
| 3 | 3 | 1.000 | 1.000 | 1.000 | 0.033 | 0.027 | True |
| 3 | 4 | 1.000 | 1.000 | 1.000 | 0.033 | 0.027 | True |
| 3 | 5 | 1.000 | 1.000 | 1.000 | 0.033 | 0.027 | True |
| 4 | 1 | 0.900 | 1.000 | 0.947 | -0.107 | -0.121 | False |
| 4 | 2 | 0.900 | 1.000 | 0.947 | -0.068 | -0.079 | False |
| 4 | 3 | 0.900 | 1.000 | 0.947 | -0.026 | -0.034 | False |
| 4 | 4 | 1.000 | 1.000 | 1.000 | 0.022 | 0.018 | True |
| 4 | 5 | 1.000 | 1.000 | 1.000 | 0.022 | 0.018 | True |
| 5 | 1 | 0.917 | 1.000 | 0.957 | -0.117 | -0.135 | False |
| 5 | 2 | 0.917 | 1.000 | 0.957 | -0.087 | -0.104 | False |
| 5 | 3 | 0.917 | 1.000 | 0.957 | -0.056 | -0.070 | False |
| 5 | 4 | 0.917 | 1.000 | 0.957 | -0.023 | -0.032 | False |
| 5 | 5 | 1.000 | 1.000 | 1.000 | 0.016 | 0.011 | True |

Required conditioning set size for clean separation:
- `1` confounders: `1`
- `2` confounders: `2`
- `3` confounders: `3`
- `4` confounders: `4`
- `5` confounders: `5`

Interpretation:
- For multi-confound worlds, failure at 4-5 confounders with smaller `k` is computational/statistical under a fixed conditioning budget, not representational.
- The signal exists when the conditioning set can include all explanatory confounders.
- With concept-level representations, current WorldField statistics can recover undirected causal skeletons on these synthetic worlds.

## Classical Baselines

The PC-style baseline is essentially an exhaustive conditional-independence pruning procedure, so it tracks the pure CMI learner closely.
Partial correlation and sparse regression are included as lightweight linear baselines; they are useful controls, not the primary claim.

## Failure Cases

- `double_confound` / `pure_cmi_k1` seed 0: {'precision': 0.8333333333333334, 'recall': 1.0, 'f1': 0.9090909090909091, 'tp': 5, 'fp': 1, 'fn': 0}
- `triple_confound` / `pure_cmi_k1` seed 0: {'precision': 0.875, 'recall': 1.0, 'f1': 0.9333333333333333, 'tp': 7, 'fp': 1, 'fn': 0}
- `triple_confound` / `pure_cmi_k2` seed 0: {'precision': 0.875, 'recall': 1.0, 'f1': 0.9333333333333333, 'tp': 7, 'fp': 1, 'fn': 0}
- `chain` / `pure_cmi_k1` seed 0: {'precision': 0.75, 'recall': 1.0, 'f1': 0.8571428571428571, 'tp': 3, 'fp': 1, 'fn': 0}
- `chain` / `pure_cmi_k2` seed 0: {'precision': 0.75, 'recall': 1.0, 'f1': 0.8571428571428571, 'tp': 3, 'fp': 1, 'fn': 0}
- `chain` / `pure_cmi_k3` seed 0: {'precision': 0.75, 'recall': 1.0, 'f1': 0.8571428571428571, 'tp': 3, 'fp': 1, 'fn': 0}

## Relation To WorldField

This operates directly on concept-level event variables. It does not yet prove fragment-scale causality.
The feasible path is: concepts first, cluster units next, fragment graphs only after the cluster-level test survives.

## Verdict

Causal graph recovery is possible with the current concept-level WorldField representations on these controlled worlds. The system is no longer limited to rejecting a single confound.
The claim should remain scoped to undirected skeleton recovery from conditional-information statistics, not causal direction or general real-world causal discovery.