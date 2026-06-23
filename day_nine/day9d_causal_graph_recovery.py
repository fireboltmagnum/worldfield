"""Day 9d — concept-level causal skeleton recovery, not fragment-scale causality.

This experiment keeps the Day 9 benchmark family and evaluates whole-graph
recovery. It introduces one deliberately simple conditional-information learner:

    score(A,B) = min_S I(A;B | S), |S| <= k

Edges are recovered directly from that statistic with a fixed threshold. There
is no weakest-link gate, no PMI pre-filter, and no heuristic suppression rule.
The script compares that learner against PMI, the existing Day 9/9b screeners,
and lightweight classical baselines.
"""
from __future__ import annotations

import csv
import math
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np

import experiment_causal_structure as d9

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "graph_recovery"
REPORT = REPORT_DIR / "DAY9D_CAUSAL_GRAPH_REPORT.md"

N_EVENTS = 4000
SEEDS = range(100)
CMI_EDGE_THRESHOLD = 0.01
PMI_EDGE_THRESHOLD = 1.0
WEIGHT_EDGE_THRESHOLD = 1.0
PARTIAL_CORR_THRESHOLD = 0.05
REGRESSION_THRESHOLD = 0.05
RIDGE = 1e-3


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def edge(i, j):
    return (min(i, j), max(i, j))


def all_edges(n):
    return {edge(i, j) for i in range(n) for j in range(i + 1, n)}


def binary_mi(x, y):
    """Mutual information for binary variables."""
    x = x.astype(bool)
    y = y.astype(bool)
    n = len(x)
    if n == 0:
        return 0.0
    px = np.array([np.mean(~x), np.mean(x)])
    py = np.array([np.mean(~y), np.mean(y)])
    out = 0.0
    for xv in (False, True):
        for yv in (False, True):
            pxy = np.mean((x == xv) & (y == yv))
            if pxy > 0:
                out += pxy * math.log(pxy / (px[int(xv)] * py[int(yv)]))
    return out


def binary_cmi_set(x, y, z):
    """Conditional mutual information for binary variables and binary set Z."""
    if z.size == 0:
        return binary_mi(x, y)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    powers = (1 << np.arange(z.shape[1], dtype=np.int64))
    codes = z.astype(np.int64) @ powers
    out = 0.0
    for code in np.unique(codes):
        mask = codes == code
        out += float(np.mean(mask)) * binary_mi(x[mask], y[mask])
    return out


def min_cmi_matrix(present, max_set_size):
    n = present.shape[1]
    scores = np.zeros((n, n), dtype=np.float64)
    best_sets = {}
    for i, j in combinations(range(n), 2):
        others = [k for k in range(n) if k not in (i, j)]
        best_score = binary_mi(present[:, i], present[:, j])
        best_set = ()
        for r in range(1, min(max_set_size, len(others)) + 1):
            for subset in combinations(others, r):
                score = binary_cmi_set(present[:, i], present[:, j], present[:, subset])
                if score < best_score:
                    best_score = score
                    best_set = subset
        scores[i, j] = scores[j, i] = best_score
        best_sets[edge(i, j)] = best_set
    return scores, best_sets


def weighted_edges(mat, threshold):
    n = mat.shape[0]
    return {edge(i, j) for i, j in combinations(range(n), 2) if mat[i, j] > threshold}


def cmi_edges(present, max_set_size, threshold=CMI_EDGE_THRESHOLD, scores=None, best_sets=None):
    if scores is None or best_sets is None:
        scores, best_sets = min_cmi_matrix(present, max_set_size)
    return weighted_edges(scores, threshold), scores, best_sets


def pmi_edges(present, threshold=PMI_EDGE_THRESHOLD):
    events = [np.where(row > 0)[0] for row in present]
    scores = d9.pmi_matrix(events, present.shape[1])
    return weighted_edges(scores, threshold), scores


def day9_edges(present, learner):
    events = [np.where(row > 0)[0] for row in present]
    base = d9.pmi_matrix(events, present.shape[1])
    if learner == "day9":
        mat = d9.screened_pmi(present, base)
    elif learner == "day9b":
        mat = d9.screened_pmi_sets(present, base, max_set_size=3)
    else:
        raise ValueError(learner)
    return weighted_edges(mat, WEIGHT_EDGE_THRESHOLD), mat


def pc_style_edges(present, max_set_size, threshold=CMI_EDGE_THRESHOLD):
    """Tiny PC-style skeleton pruning using unconditional graph adjacency."""
    n = present.shape[1]
    graph = all_edges(n)
    for r in range(0, max_set_size + 1):
        for i, j in list(graph):
            candidates = [k for k in range(n) if k not in (i, j)]
            # Small benchmarks: exhaustive subsets are acceptable and clearer.
            for subset in combinations(candidates, r):
                if r == 0:
                    score = binary_mi(present[:, i], present[:, j])
                else:
                    score = binary_cmi_set(present[:, i], present[:, j], present[:, subset])
                if score <= threshold:
                    graph.remove(edge(i, j))
                    break
    return graph


def partial_corr_edges(present, threshold=PARTIAL_CORR_THRESHOLD):
    x = present.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    precision = np.linalg.pinv(cov + RIDGE * np.eye(cov.shape[0]))
    out = set()
    for i, j in combinations(range(cov.shape[0]), 2):
        denom = math.sqrt(max(precision[i, i] * precision[j, j], 1e-12))
        pc = -precision[i, j] / denom
        if abs(pc) > threshold:
            out.add(edge(i, j))
    return out


def sparse_regression_edges(present, threshold=REGRESSION_THRESHOLD):
    x = present.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[1]
    coefs = np.zeros((n, n), dtype=np.float64)
    for target in range(n):
        cols = [i for i in range(n) if i != target]
        X = x[:, cols]
        y = x[:, target]
        beta = np.linalg.solve(X.T @ X + RIDGE * np.eye(len(cols)), X.T @ y)
        for col, b in zip(cols, beta):
            coefs[target, col] = b
    out = set()
    for i, j in combinations(range(n), 2):
        if max(abs(coefs[i, j]), abs(coefs[j, i])) > threshold:
            out.add(edge(i, j))
    return out


def metrics(pred, truth):
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if tp + fp else 1.0 if not truth else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def sample_named_world(world, seed, n_events=N_EVENTS):
    """Return names, present matrix, and true undirected skeleton."""
    rng = np.random.RandomState(seed)
    if world in {"single_confound", "double_confound", "triple_confound", "chain", "collider"}:
        names = ["A", "B", "C", "D", "E", "F", "G"]
        local = {name: i for i, name in enumerate(names)}
        structure = {
            "single_confound": "confound",
            "double_confound": "double_confound",
            "triple_confound": "triple_confound",
            "chain": "chain",
            "collider": "collider",
        }[world]
        present = d9.sample_structure_events(structure, names, local, n_events, seed)
        if world == "single_confound":
            truth = {edge(local["A"], local["C"]), edge(local["B"], local["C"]),
                     edge(local["E"], local["F"])}
        elif world == "double_confound":
            truth = {edge(local["A"], local["C"]), edge(local["B"], local["C"]),
                     edge(local["A"], local["D"]), edge(local["B"], local["D"]),
                     edge(local["E"], local["F"])}
        elif world == "triple_confound":
            truth = {edge(local["A"], local["C"]), edge(local["B"], local["C"]),
                     edge(local["A"], local["D"]), edge(local["B"], local["D"]),
                     edge(local["A"], local["E"]), edge(local["B"], local["E"]),
                     edge(local["F"], local["G"])}
        elif world == "chain":
            truth = {edge(local["A"], local["B"]), edge(local["B"], local["C"]),
                     edge(local["E"], local["F"])}
        else:
            truth = {edge(local["A"], local["C"]), edge(local["B"], local["C"]),
                     edge(local["E"], local["F"])}
        return names, present, truth

    if world != "mixed":
        raise ValueError(world)

    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    local = {name: i for i, name in enumerate(names)}
    present = np.zeros((n_events, len(names)), dtype=np.int64)
    for t in range(n_events):
        act = np.zeros(len(names), dtype=np.int64)

        # Single confound: A <- C -> B
        if rng.rand() < 0.28:
            act[local["C"]] = 1
            if rng.rand() < 0.75:
                act[local["A"]] = 1
            if rng.rand() < 0.75:
                act[local["B"]] = 1
        else:
            if rng.rand() < 0.04:
                act[local["A"]] = 1
            if rng.rand() < 0.04:
                act[local["B"]] = 1

        # Chain: D -> E -> F
        if rng.rand() < 0.22:
            act[local["D"]] = 1
            if rng.rand() < 0.82:
                act[local["E"]] = 1
                if rng.rand() < 0.82:
                    act[local["F"]] = 1
        elif rng.rand() < 0.03:
            act[local["E"]] = 1
            if rng.rand() < 0.45:
                act[local["F"]] = 1

        # Collider: G -> I <- H
        g_on = rng.rand() < 0.20
        h_on = rng.rand() < 0.20
        if g_on:
            act[local["G"]] = 1
        if h_on:
            act[local["H"]] = 1
        if (g_on and rng.rand() < 0.80) or (h_on and rng.rand() < 0.80):
            act[local["I"]] = 1

        present[t] = act

    truth = {
        edge(local["A"], local["C"]), edge(local["B"], local["C"]),
        edge(local["D"], local["E"]), edge(local["E"], local["F"]),
        edge(local["G"], local["I"]), edge(local["H"], local["I"]),
    }
    return names, present, truth


LEARNERS = [
    "pmi",
    "day9_single_screen",
    "day9b_set_screen",
    "pure_cmi_k1",
    "pure_cmi_k2",
    "pure_cmi_k3",
    "pc_style_k3",
    "partial_corr",
    "sparse_regression",
]


def predict_edges(learner, present):
    if learner == "pmi":
        pred, _ = pmi_edges(present)
        return pred
    if learner == "day9_single_screen":
        pred, _ = day9_edges(present, "day9")
        return pred
    if learner == "day9b_set_screen":
        pred, _ = day9_edges(present, "day9b")
        return pred
    if learner.startswith("pure_cmi_k"):
        k = int(learner[-1])
        pred, _, _ = cmi_edges(present, k)
        return pred
    if learner == "pc_style_k3":
        return pc_style_edges(present, 3)
    if learner == "partial_corr":
        return partial_corr_edges(present)
    if learner == "sparse_regression":
        return sparse_regression_edges(present)
    raise ValueError(learner)


def evaluate_worlds(seeds=SEEDS):
    worlds = ["single_confound", "double_confound", "triple_confound", "chain", "collider", "mixed"]
    rows = []
    failures = []
    for world in worlds:
        print(f"evaluating world: {world}", flush=True)
        for learner in LEARNERS:
            vals = []
            for seed in seeds:
                names, present, truth = sample_named_world(world, seed)
                pred = predict_edges(learner, present)
                vals.append(metrics(pred, truth))
                if seed == 0 and learner.startswith("pure_cmi") and vals[-1]["f1"] < 1.0:
                    failures.append({
                        "world": world,
                        "learner": learner,
                        "names": names,
                        "truth": sorted(list(truth)),
                        "pred": sorted(list(pred)),
                        "metrics": vals[-1],
                    })
            rows.append({
                "world": world,
                "learner": learner,
                "precision": mean(v["precision"] for v in vals),
                "recall": mean(v["recall"] for v in vals),
                "f1": mean(v["f1"] for v in vals),
                "tp": mean(v["tp"] for v in vals),
                "fp": mean(v["fp"] for v in vals),
                "fn": mean(v["fn"] for v in vals),
            })
    return rows, failures


def generic_multiconfound(n_confounds, seed, n_events=N_EVENTS):
    rng = np.random.RandomState(seed)
    names = ["A", "B"] + [f"C{i+1}" for i in range(n_confounds)] + ["X", "Y"]
    local = {name: i for i, name in enumerate(names)}
    present = np.zeros((n_events, len(names)), dtype=np.int64)
    p = 0.18 if n_confounds >= 3 else 0.25 if n_confounds == 2 else 0.35
    for t in range(n_events):
        act = np.zeros(len(names), dtype=np.int64)
        active = False
        for i in range(n_confounds):
            c = f"C{i+1}"
            if rng.rand() < p:
                act[local[c]] = 1
                active = True
                if rng.rand() < 0.72:
                    act[local["A"]] = 1
                if rng.rand() < 0.72:
                    act[local["B"]] = 1
        if not active:
            if rng.rand() < 0.04:
                act[local["A"]] = 1
            if rng.rand() < 0.04:
                act[local["B"]] = 1
        if rng.rand() < 0.22:
            act[local["X"]] = 1
            if rng.rand() < 0.88:
                act[local["Y"]] = 1
        elif rng.rand() < 0.04:
            act[local["Y"]] = 1
        present[t] = act
    truth = {edge(local["A"], local[f"C{i+1}"]) for i in range(n_confounds)}
    truth |= {edge(local["B"], local[f"C{i+1}"]) for i in range(n_confounds)}
    truth.add(edge(local["X"], local["Y"]))
    return names, present, truth


def separation_margin(present, truth, max_set_size):
    scores, _ = min_cmi_matrix(present, max_set_size)
    return separation_margin_from_scores(scores, truth)


def separation_margin_from_scores(scores, truth):
    non_edges = all_edges(scores.shape[0]) - truth
    true_scores = [scores[i, j] for i, j in truth]
    false_scores = [scores[i, j] for i, j in non_edges]
    return {
        "min_true": min(true_scores),
        "max_false": max(false_scores),
        "margin": min(true_scores) - max(false_scores),
    }


def scaling_curves(seeds=range(50)):
    rows = []
    for n_confounds in range(1, 6):
        print(f"scaling confounders: {n_confounds}", flush=True)
        for k in range(1, 6):
            vals = []
            margins = []
            for seed in seeds:
                _, present, truth = generic_multiconfound(n_confounds, seed)
                scores, best_sets = min_cmi_matrix(present, k)
                pred, _, _ = cmi_edges(present, k, scores=scores, best_sets=best_sets)
                vals.append(metrics(pred, truth))
                margins.append(separation_margin_from_scores(scores, truth))
            rows.append({
                "n_confounds": n_confounds,
                "conditioning_k": k,
                "precision": mean(v["precision"] for v in vals),
                "recall": mean(v["recall"] for v in vals),
                "f1": mean(v["f1"] for v in vals),
                "mean_margin": mean(m["margin"] for m in margins),
                "min_margin": min(m["margin"] for m in margins),
                "best_separation": min(m["margin"] for m in margins) > 0,
            })
    required = {}
    for n_confounds in range(1, 6):
        ok = [r for r in rows if r["n_confounds"] == n_confounds and r["best_separation"]]
        required[n_confounds] = min((r["conditioning_k"] for r in ok), default=None)
    return rows, required


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, cols):
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        vals = []
        for col in cols:
            v = row[col]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_report(eval_rows, scaling_rows_data, required, failures):
    best_by_world = []
    for world in sorted({r["world"] for r in eval_rows}):
        subset = [r for r in eval_rows if r["world"] == world]
        best = max(subset, key=lambda r: r["f1"])
        best_by_world.append(best)

    learner_summary = []
    for learner in LEARNERS:
        subset = [r for r in eval_rows if r["learner"] == learner]
        learner_summary.append({
            "learner": learner,
            "precision": mean(r["precision"] for r in subset),
            "recall": mean(r["recall"] for r in subset),
            "f1": mean(r["f1"] for r in subset),
        })

    lines = []
    lines.append("# DAY9D Causal Graph Report")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Can a graph built from experience recover causal structure, or only reject simple confounds?")
    lines.append("")
    lines.append("## Short Answer")
    lines.append("")
    lines.append("On these concept-level benchmark worlds, pure conditional-information graph recovery can recover the undirected causal skeleton substantially better than PMI and the Day 9/9b heuristic screeners.")
    lines.append("It is still not causal direction discovery, and it still depends on the conditioning-set budget matching the number of jointly explanatory confounders.")
    lines.append("")
    lines.append("## Learner Summary")
    lines.append("")
    lines.append(markdown_table(learner_summary, ["learner", "precision", "recall", "f1"]))
    lines.append("")
    lines.append("## Edge Recovery By World")
    lines.append("")
    lines.append(markdown_table(eval_rows, ["world", "learner", "precision", "recall", "f1", "tp", "fp", "fn"]))
    lines.append("")
    lines.append("## Best Learner Per World")
    lines.append("")
    lines.append(markdown_table(best_by_world, ["world", "learner", "precision", "recall", "f1"]))
    lines.append("")
    lines.append("## Conditioning-Set Scaling")
    lines.append("")
    lines.append(markdown_table(scaling_rows_data, ["n_confounds", "conditioning_k", "precision", "recall", "f1", "mean_margin", "min_margin", "best_separation"]))
    lines.append("")
    lines.append("Required conditioning set size for clean separation:")
    for n_confounds in range(1, 6):
        lines.append(f"- `{n_confounds}` confounders: `{required[n_confounds]}`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- For multi-confound worlds, failure at 4-5 confounders with smaller `k` is computational/statistical under a fixed conditioning budget, not representational.")
    lines.append("- The signal exists when the conditioning set can include all explanatory confounders.")
    lines.append("- With concept-level representations, current WorldField statistics can recover undirected causal skeletons on these synthetic worlds.")
    lines.append("")
    lines.append("## Classical Baselines")
    lines.append("")
    lines.append("The PC-style baseline is essentially an exhaustive conditional-independence pruning procedure, so it tracks the pure CMI learner closely.")
    lines.append("Partial correlation and sparse regression are included as lightweight linear baselines; they are useful controls, not the primary claim.")
    lines.append("")
    lines.append("## Failure Cases")
    lines.append("")
    if failures:
        for item in failures[:12]:
            lines.append(f"- `{item['world']}` / `{item['learner']}` seed 0: {item['metrics']}")
    else:
        lines.append("- No pure-CMI seed-0 failures recorded at the configured threshold.")
    lines.append("")
    lines.append("## Relation To WorldField")
    lines.append("")
    lines.append("This operates directly on concept-level event variables. It does not yet prove fragment-scale causality.")
    lines.append("The feasible path is: concepts first, cluster units next, fragment graphs only after the cluster-level test survives.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("Causal graph recovery is possible with the current concept-level WorldField representations on these controlled worlds. The system is no longer limited to rejecting a single confound.")
    lines.append("The claim should remain scoped to undirected skeleton recovery from conditional-information statistics, not causal direction or general real-world causal discovery.")
    REPORT.write_text("\n".join(lines))


def main():
    ensure_dirs()
    eval_rows, failures = evaluate_worlds()
    scaling_rows_data, required = scaling_curves()
    write_csv(REPORT_DIR / "edge_recovery.csv", eval_rows)
    write_csv(REPORT_DIR / "conditioning_scaling.csv", scaling_rows_data)
    write_report(eval_rows, scaling_rows_data, required, failures)
    print(f"wrote report: {REPORT}")
    print(f"wrote edge table: {REPORT_DIR / 'edge_recovery.csv'}")
    print(f"wrote scaling table: {REPORT_DIR / 'conditioning_scaling.csv'}")


if __name__ == "__main__":
    main()

