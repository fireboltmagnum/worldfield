"""Day 9e — scalability study for the pure CMI learner.

This does not add a new learner. Exact recovery uses the Day 9d pure CMI rule:

    score(A,B) = min_S I(A;B | S), |S| <= k

The study separates exact executions from projected executions. For large
concept spaces, exhaustive CMI is not silently approximated; the script records
the exact conditioning-test count, memory estimate, measured per-test cost, and
why the full run was not attempted.
"""
from __future__ import annotations

import csv
import math
import resource
import time
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np

import day9d_causal_graph_recovery as d9d

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "scalability"
REPORT = REPORT_DIR / "DAY9E_SCALABILITY_REPORT.md"

THRESHOLD = d9d.CMI_EDGE_THRESHOLD
DEFAULT_K = 3
DEFAULT_EVENTS = 4000
EXACT_TEST_LIMIT = 900_000
EXACT_WALL_LIMIT_SEC = 120.0


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def rss_mb():
    # macOS reports ru_maxrss in bytes; Linux reports KB. This repo is on macOS.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024)


def edge(i, j):
    return d9d.edge(i, j)


def conditioning_tests(n_concepts, k):
    total_per_pair = 1  # unconditional MI baseline
    for r in range(1, min(k, n_concepts - 2) + 1):
        total_per_pair += math.comb(n_concepts - 2, r)
    return math.comb(n_concepts, 2) * total_per_pair


def approx_memory_mb(n_events, n_concepts):
    present = n_events * n_concepts * 8
    scores = n_concepts * n_concepts * 8
    # Include room for Python containers, temporary masks/codes, and report rows.
    return (present + scores) / (1024 * 1024) * 3.0


def metrics(pred, truth):
    return d9d.metrics(pred, truth)


def recover_exact(present, truth, k):
    start = time.perf_counter()
    start_rss = rss_mb()
    scores, best_sets = d9d.min_cmi_matrix(present, k)
    pred, _, _ = d9d.cmi_edges(present, k, scores=scores, best_sets=best_sets)
    elapsed = time.perf_counter() - start
    out = metrics(pred, truth)
    out.update({
        "runtime_sec": elapsed,
        "rss_peak_mb": rss_mb(),
        "rss_delta_mb": max(0.0, rss_mb() - start_rss),
        "pred_edges": len(pred),
        "truth_edges": len(truth),
    })
    return out


def sample_scaling_world(n_concepts, n_events, seed, distractor_rate=0.01,
                         base_rate=0.22, effect=0.76):
    """Mixed causal skeleton embedded in a larger concept space."""
    if n_concepts < 9:
        raise ValueError("scaling world needs at least 9 concepts")
    rng = np.random.RandomState(seed)
    present = np.zeros((n_events, n_concepts), dtype=np.int64)
    truth = {
        edge(0, 2), edge(1, 2),        # single confound: A <- C -> B
        edge(3, 4), edge(4, 5),        # chain: D -> E -> F
        edge(6, 8), edge(7, 8),        # collider: G -> I <- H
    }
    for t in range(n_events):
        act = np.zeros(n_concepts, dtype=np.int64)

        if rng.rand() < base_rate:
            act[2] = 1
            if rng.rand() < effect:
                act[0] = 1
            if rng.rand() < effect:
                act[1] = 1
        else:
            if rng.rand() < 0.03:
                act[0] = 1
            if rng.rand() < 0.03:
                act[1] = 1

        if rng.rand() < base_rate:
            act[3] = 1
            if rng.rand() < effect:
                act[4] = 1
                if rng.rand() < effect:
                    act[5] = 1
        elif rng.rand() < 0.03:
            act[4] = 1
            if rng.rand() < 0.40:
                act[5] = 1

        g_on = rng.rand() < base_rate
        h_on = rng.rand() < base_rate
        if g_on:
            act[6] = 1
        if h_on:
            act[7] = 1
        if (g_on and rng.rand() < effect) or (h_on and rng.rand() < effect):
            act[8] = 1

        if n_concepts > 9:
            act[9:] = rng.rand(n_concepts - 9) < distractor_rate
        present[t] = act
    return present, truth


def sample_multiconfound_world(n_confounds, n_events, seed, p=None):
    rng = np.random.RandomState(seed)
    n = 2 + n_confounds + 2
    present = np.zeros((n_events, n), dtype=np.int64)
    if p is None:
        p = 0.35 if n_confounds == 1 else 0.25 if n_confounds == 2 else 0.16
    for t in range(n_events):
        act = np.zeros(n, dtype=np.int64)
        active = False
        for c in range(2, 2 + n_confounds):
            if rng.rand() < p:
                act[c] = 1
                active = True
                if rng.rand() < 0.72:
                    act[0] = 1
                if rng.rand() < 0.72:
                    act[1] = 1
        if not active:
            if rng.rand() < 0.04:
                act[0] = 1
            if rng.rand() < 0.04:
                act[1] = 1
        if rng.rand() < 0.22:
            act[-2] = 1
            if rng.rand() < 0.88:
                act[-1] = 1
        elif rng.rand() < 0.04:
            act[-1] = 1
        present[t] = act
    truth = {edge(0, c) for c in range(2, 2 + n_confounds)}
    truth |= {edge(1, c) for c in range(2, 2 + n_confounds)}
    truth.add(edge(n - 2, n - 1))
    return present, truth


def probe_test_cost(n_events=1000, n_concepts=12, n_tests=2500, seed=0):
    rng = np.random.RandomState(seed)
    present = rng.rand(n_events, n_concepts) < 0.08
    pairs = list(combinations(range(n_concepts), 2))
    start = time.perf_counter()
    for i in range(n_tests):
        a, b = pairs[i % len(pairs)]
        others = [x for x in range(n_concepts) if x not in (a, b)]
        subset = tuple(rng.choice(others, size=3, replace=False))
        d9d.binary_cmi_set(present[:, a], present[:, b], present[:, subset])
    elapsed = time.perf_counter() - start
    return elapsed / n_tests


def maybe_exact_row(experiment, n_concepts, n_events, k, seed, truth_name,
                    distractor_rate=0.01):
    tests = conditioning_tests(n_concepts, k)
    row = {
        "experiment": experiment,
        "case": truth_name,
        "concepts": n_concepts,
        "events": n_events,
        "k": k,
        "conditioning_tests": tests,
        "precision": "not_run",
        "recall": "not_run",
        "f1": "not_run",
        "runtime_sec": "not_run",
        "rss_peak_mb": "not_run",
        "rss_delta_mb": "not_run",
        "status": "projected_only",
        "reason": "conditioning_tests_exceed_limit",
        "memory_estimate_mb": approx_memory_mb(n_events, n_concepts),
    }
    if tests > EXACT_TEST_LIMIT:
        return row

    present, truth = sample_scaling_world(
        n_concepts, n_events, seed, distractor_rate=distractor_rate
    )
    start = time.perf_counter()
    exact = recover_exact(present, truth, k)
    if exact["runtime_sec"] > EXACT_WALL_LIMIT_SEC:
        row["reason"] = "completed_but_exceeded_wall_limit"
    else:
        row["reason"] = "completed_exact"
    row.update({
        "precision": exact["precision"],
        "recall": exact["recall"],
        "f1": exact["f1"],
        "runtime_sec": exact["runtime_sec"],
        "rss_peak_mb": exact["rss_peak_mb"],
        "rss_delta_mb": exact["rss_delta_mb"],
        "status": "exact",
    })
    row["wall_sec_including_generation"] = time.perf_counter() - start
    return row


def concept_count_scaling(sec_per_test):
    rows = []
    for n in [60, 120, 240, 480, 960]:
        row = maybe_exact_row("concept_count", n, DEFAULT_EVENTS, DEFAULT_K, 10, f"{n}_concepts")
        row["projected_runtime_sec"] = conditioning_tests(n, DEFAULT_K) * sec_per_test
        rows.append(row)
        print(f"concept count row: {n}", flush=True)
    return rows


def distractor_scaling(sec_per_test):
    rows = []
    for extra in [50, 100, 250, 500, 1000]:
        n = 9 + extra
        row = maybe_exact_row("distractor", n, DEFAULT_EVENTS, DEFAULT_K, 20, f"+{extra}")
        row["distractors"] = extra
        row["projected_runtime_sec"] = conditioning_tests(n, DEFAULT_K) * sec_per_test
        rows.append(row)
        print(f"distractor row: +{extra}", flush=True)
    return rows


def sample_complexity():
    rows = []
    event_counts = [100, 500, 1000, 4000, 10000, 50000]
    for n_events in event_counts:
        seed_count = 20 if n_events <= 10_000 else 5
        vals = []
        runtimes = []
        rss = []
        for seed in range(seed_count):
            present, truth = sample_scaling_world(9, n_events, seed + 100)
            exact = recover_exact(present, truth, DEFAULT_K)
            vals.append(exact)
            runtimes.append(exact["runtime_sec"])
            rss.append(exact["rss_peak_mb"])
        rows.append({
            "experiment": "sample_complexity",
            "case": f"{n_events}_events",
            "concepts": 9,
            "events": n_events,
            "k": DEFAULT_K,
            "seeds": seed_count,
            "conditioning_tests": conditioning_tests(9, DEFAULT_K),
            "precision": mean(v["precision"] for v in vals),
            "recall": mean(v["recall"] for v in vals),
            "f1": mean(v["f1"] for v in vals),
            "runtime_sec": mean(runtimes),
            "rss_peak_mb": max(rss),
            "status": "exact",
        })
        print(f"sample complexity row: {n_events}", flush=True)
    return rows


def conditioning_explosion():
    rows = []
    for confounds in range(1, 9):
        vals = []
        runtimes = []
        rss = []
        n_concepts = 2 + confounds + 2
        k = confounds
        for seed in range(5):
            present, truth = sample_multiconfound_world(confounds, DEFAULT_EVENTS, seed + 200)
            exact = recover_exact(present, truth, k)
            vals.append(exact)
            runtimes.append(exact["runtime_sec"])
            rss.append(exact["rss_peak_mb"])
        rows.append({
            "experiment": "conditioning_explosion",
            "case": f"{confounds}_confounders",
            "confounders": confounds,
            "concepts": n_concepts,
            "events": DEFAULT_EVENTS,
            "k": k,
            "conditioning_tests": conditioning_tests(n_concepts, k),
            "precision": mean(v["precision"] for v in vals),
            "recall": mean(v["recall"] for v in vals),
            "f1": mean(v["f1"] for v in vals),
            "runtime_sec": mean(runtimes),
            "rss_peak_mb": max(rss),
            "status": "exact",
        })
        print(f"conditioning row: {confounds}", flush=True)
    return rows


def sparsity_probe():
    rows = []
    for rate in [0.001, 0.005, 0.01, 0.03, 0.08]:
        vals = []
        for seed in range(10):
            present, truth = sample_scaling_world(
                30, DEFAULT_EVENTS, seed + 300, distractor_rate=rate
            )
            # k=1 is the largest exact run this sparsity sweep can afford while
            # preserving exhaustive pure CMI over all pairs.
            exact = recover_exact(present, truth, 1)
            vals.append(exact)
        rows.append({
            "experiment": "event_sparsity",
            "case": f"distractor_rate_{rate}",
            "concepts": 30,
            "events": DEFAULT_EVENTS,
            "k": 1,
            "conditioning_tests": conditioning_tests(30, 1),
            "precision": mean(v["precision"] for v in vals),
            "recall": mean(v["recall"] for v in vals),
            "f1": mean(v["f1"] for v in vals),
            "runtime_sec": mean(v["runtime_sec"] for v in vals),
            "rss_peak_mb": max(v["rss_peak_mb"] for v in vals),
            "status": "exact",
        })
        print(f"sparsity row: {rate}", flush=True)
    return rows


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def human_seconds(seconds):
    if not isinstance(seconds, (float, int)):
        return str(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    if seconds < 31_536_000:
        return f"{seconds / 86400:.1f}d"
    return f"{seconds / 31_536_000:.1f}y"


def write_report(sec_per_test, concept_rows, distractor_rows, sample_rows,
                 conditioning_rows, sparsity_rows):
    concept_display = []
    for row in concept_rows:
        out = dict(row)
        out["projected_runtime"] = human_seconds(row["projected_runtime_sec"])
        concept_display.append(out)

    distractor_display = []
    for row in distractor_rows:
        out = dict(row)
        out["projected_runtime"] = human_seconds(row["projected_runtime_sec"])
        distractor_display.append(out)

    reliable = [r for r in sample_rows if isinstance(r["f1"], float) and r["f1"] >= 0.95]
    min_reliable = min((r["events"] for r in reliable), default=None)

    max_exact = max(
        [r["concepts"] for r in concept_rows if r["status"] == "exact"] + [9],
        default=9,
    )
    feasible_60 = concept_rows[0]["status"] == "exact"

    lines = [
        "# DAY9E Scalability Report",
        "",
        "## Question",
        "",
        "Does the Day 9d pure CMI learner survive scale?",
        "",
        "## Method",
        "",
        "The learner was not changed. Exact runs use `score(A,B) = min I(A;B|S)` over exhaustive conditioning sets up to `k`.",
        "Large concept-space cases are marked `projected_only` when the exact conditioning-test count exceeds the run limit; these are not approximated recovery results.",
        f"Measured probe cost: `{sec_per_test:.8f}` seconds per CMI test on this machine.",
        "",
        "## Concept Count Scaling",
        "",
        md_table(concept_display, [
            "case", "concepts", "events", "k", "conditioning_tests",
            "status", "projected_runtime", "memory_estimate_mb", "f1", "reason",
        ]),
        "",
        "## Distractor Scaling",
        "",
        md_table(distractor_display, [
            "case", "distractors", "concepts", "events", "k",
            "conditioning_tests", "status", "projected_runtime", "f1", "reason",
        ]),
        "",
        "## Sample Complexity",
        "",
        md_table(sample_rows, [
            "case", "events", "seeds", "precision", "recall", "f1",
            "runtime_sec", "rss_peak_mb",
        ]),
        "",
        f"Minimum tested event count with mean F1 >= 0.95: `{min_reliable}`.",
        "",
        "## Conditioning Explosion",
        "",
        md_table(conditioning_rows, [
            "case", "confounders", "concepts", "k", "conditioning_tests",
            "precision", "recall", "f1", "runtime_sec", "rss_peak_mb",
        ]),
        "",
        "## Event Sparsity",
        "",
        md_table(sparsity_rows, [
            "case", "concepts", "events", "k", "conditioning_tests",
            "precision", "recall", "f1", "runtime_sec", "rss_peak_mb",
        ]),
        "",
        "## Answers",
        "",
        f"- Does recovery survive larger concept spaces? Not with exhaustive pure CMI at `k=3`. The requested 60+ concept runs exceed the exact-run budget before recovery quality is the limiting factor.",
        "- What breaks first? Computation breaks first: the number of conditioning tests grows combinatorially with concept count and `k`.",
        "- Statistical failure or computational failure? For the tested toy skeleton, sample complexity is manageable at small concept count; large-space failure is computational before it is statistical.",
        f"- Estimated maximum feasible concept count on current hardware: about `{max_exact}` concepts for exact full-matrix `k=3` under this script's conservative interactive limit; 60 concepts already projects beyond practical interactive runtime.",
        "- Is CMI worth continuing? Yes as a diagnostic/statistical direction, but not as a naive all-pairs exhaustive learner.",
        "- Next bottleneck after confound rejection: candidate-set control. The project needs a principled way to restrict which pairs and conditioning variables are tested without reintroducing the Day 9b heuristic failure.",
        "",
        "## Verdict",
        "",
        "Day 9e changes the interpretation of Day 9d: pure CMI works as a clean causal-skeleton statistic, but exhaustive graph recovery does not scale to realistic concept spaces.",
        "The next research problem is not another learner gate; it is scalable candidate generation that preserves the valid conditioning sets Day 9c/9d proved are necessary.",
    ]
    if not feasible_60:
        lines.append("")
        lines.append("The 60-concept requested case was intentionally not approximated as a pass/fail recovery run; the report records it as a computational scaling failure.")
    REPORT.write_text("\n".join(lines))


def main():
    ensure_dirs()
    sec_per_test = probe_test_cost()
    print(f"probe seconds per CMI test: {sec_per_test:.8f}", flush=True)
    concept_rows = concept_count_scaling(sec_per_test)
    distractor_rows = distractor_scaling(sec_per_test)
    sample_rows = sample_complexity()
    conditioning_rows = conditioning_explosion()
    sparsity_rows = sparsity_probe()

    write_csv(REPORT_DIR / "concept_count_scaling.csv", concept_rows)
    write_csv(REPORT_DIR / "distractor_scaling.csv", distractor_rows)
    write_csv(REPORT_DIR / "sample_complexity.csv", sample_rows)
    write_csv(REPORT_DIR / "conditioning_explosion.csv", conditioning_rows)
    write_csv(REPORT_DIR / "event_sparsity.csv", sparsity_rows)
    write_report(sec_per_test, concept_rows, distractor_rows, sample_rows,
                 conditioning_rows, sparsity_rows)
    print(f"wrote report: {REPORT}")


if __name__ == "__main__":
    main()
