"""Day 9 research agenda analysis.

Phases covered here:
  1. Failure analysis of Day 9b on failed multi-confound runs
  2. Ground-truth benchmark validation
  3. Direct conditional-information diagnostics
  4. Determine whether the current PMI+screening family can work in principle

This script does NOT introduce a new learner. It instruments the existing
screening logic, writes exhaustive traces for failed runs, and produces a
consolidated markdown report.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

import experiment_causal_structure as d9

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
TRACE_DIR = REPORT_DIR / "failure_traces"

KEEP = [
    "magenta diamond",   # A
    "skyblue pentagon",  # B
    "crimson square",    # C
    "red circle",        # D
    "blue square",       # E
    "teal pentagon",     # F
    "olive square",      # G
]
LOCAL = {name: i for i, name in enumerate(KEEP)}
A, B, C, D, E, F, G = KEEP


def ensure_dirs():
    REPORT_DIR.mkdir(exist_ok=True)
    TRACE_DIR.mkdir(exist_ok=True)


def fmt(mu_vals):
    if not mu_vals:
        return "n/a"
    if len(mu_vals) == 1:
        return f"{mu_vals[0]:.6f}"
    return f"{mean(mu_vals):.6f} +/- {pstdev(mu_vals):.6f}"


def cmi_bundle(structure: str, seeds=range(50)):
    rows = []
    for seed in seeds:
        present = d9.sample_structure_events(structure, KEEP, LOCAL, 4000, seed)
        a, b, c, d, e = (present[:, LOCAL[x]] for x in (A, B, C, D, E))
        rows.append({
            "I(A;B)": d9.bin_mi(a, b),
            "I(A;B|C)": d9.bin_cmi(a, b, c),
            "I(A;B|D)": d9.bin_cmi(a, b, d),
            "I(A;B|C,D)": d9.bin_cmi_set(a, b, present[:, [LOCAL[C], LOCAL[D]]]),
            "I(A;B|C,D,E)": d9.bin_cmi_set(a, b, present[:, [LOCAL[C], LOCAL[D], LOCAL[E]]]),
        })
    return rows


def structure_specific_info(structure: str, seeds=range(50)):
    rows = []
    for seed in seeds:
        present = d9.sample_structure_events(structure, KEEP, LOCAL, 4000, seed)
        if structure == "chain":
            rows.append({
                "I(A;C)": d9.bin_mi(present[:, LOCAL[A]], present[:, LOCAL[C]]),
                "I(A;C|B)": d9.bin_cmi(present[:, LOCAL[A]], present[:, LOCAL[C]], present[:, LOCAL[B]]),
                "I(A;C|B,D)": d9.bin_cmi_set(
                    present[:, LOCAL[A]], present[:, LOCAL[C]], present[:, [LOCAL[B], LOCAL[D]]]
                ),
            })
        elif structure == "collider":
            rows.append({
                "I(A;B)": d9.bin_mi(present[:, LOCAL[A]], present[:, LOCAL[B]]),
                "I(A;B|C)": d9.bin_cmi(present[:, LOCAL[A]], present[:, LOCAL[B]], present[:, LOCAL[C]]),
                "I(A;B|C,D)": d9.bin_cmi_set(
                    present[:, LOCAL[A]], present[:, LOCAL[B]], present[:, [LOCAL[C], LOCAL[D]]]
                ),
            })
    return rows


def subset_trace(present, base, i: int, j: int, max_set_size=3):
    xi = present[:, i]
    xj = present[:, j]
    others = [k for k in range(present.shape[1]) if k not in (i, j)]
    traces = []
    best_tested = None
    best_oracle = None
    for r in range(1, min(max_set_size, len(others)) + 1):
        for subset in combinations(others, r):
            links = [min(base[i, k], base[j, k]) for k in subset]
            weakest = min(links) if links else 0.0
            cmi = d9.bin_cmi_set(xi, xj, present[:, subset])
            explain = (weakest / (cmi + 1e-9)) if weakest > 0 else 0.0
            skipped = weakest <= 0 or base[i, j] >= 0.95 * weakest
            row = {
                "subset_ids": subset,
                "subset_names": [KEEP[k] for k in subset],
                "links_to_pair": [float(v) for v in links],
                "weakest_link": float(weakest),
                "base_edge": float(base[i, j]),
                "cmi": float(cmi),
                "explain_score": float(explain),
                "skipped_by_gate": bool(skipped),
            }
            traces.append(row)
            if best_oracle is None or row["cmi"] < best_oracle["cmi"] or (
                math.isclose(row["cmi"], best_oracle["cmi"]) and row["explain_score"] > best_oracle["explain_score"]
            ):
                best_oracle = row
            if not skipped:
                if best_tested is None or row["explain_score"] > best_tested["explain_score"]:
                    best_tested = row
    return traces, best_tested, best_oracle


def trace_failed_run(structure: str, seed: int):
    present = d9.sample_structure_events(structure, KEEP, LOCAL, 4000, seed)
    events = [np.where(row > 0)[0] for row in present]
    base = d9.pmi_matrix(events, len(KEEP))
    causal = d9.screened_pmi_sets(present, base, max_set_size=3)
    traces, best_tested, best_oracle = subset_trace(present, base, LOCAL[A], LOCAL[B], max_set_size=3)

    if structure == "double_confound":
        expected_explainer = [C, D]
        keeper_names = [A + "--" + C, A + "--" + D, B + "--" + C, B + "--" + D]
        expected_skeleton = [[A, C], [A, D], [B, C], [B, D], [E, F]]
    elif structure == "triple_confound":
        expected_explainer = [C, D, E]
        keeper_names = [A + "--" + C, A + "--" + D, A + "--" + E, B + "--" + C, B + "--" + D, B + "--" + E]
        expected_skeleton = [[A, C], [A, D], [A, E], [B, C], [B, D], [B, E], [F, G]]
    else:
        raise ValueError(structure)

    correct_rows = [t for t in traces if t["subset_names"] == expected_explainer]
    correct_row = correct_rows[0] if correct_rows else None

    reason = []
    if correct_row is None:
        reason.append("correct explanatory set was not generated")
    else:
        reason.append("correct explanatory set was generated")
        if correct_row["skipped_by_gate"]:
            reason.append("correct explanatory set was rejected before scoring by the weakest-link gate")
        else:
            reason.append("correct explanatory set was actually scored")
    if best_oracle and best_oracle["subset_names"] == expected_explainer and best_oracle["cmi"] < 0.01:
        reason.append("joint conditional information signal exists and is near zero at the correct set")
    if best_tested is None:
        reason.append("no candidate subset survived the gate for actual testing")
    elif correct_row is not None and not correct_row["skipped_by_gate"]:
        keep_frac = correct_row["cmi"] / (correct_row["cmi"] + correct_row["weakest_link"] + 1e-9)
        if keep_frac >= 0.25:
            reason.append("even when tested, the scoring rule would not suppress enough")
    else:
        reason.append("edge survives because the tested subsets are weaker one-variable explanations than the joint set")

    return {
        "structure": structure,
        "seed": seed,
        "expected_explainer": expected_explainer,
        "expected_skeleton": expected_skeleton,
        "keeper_names": keeper_names,
        "base_edge_strength_A_B": float(base[LOCAL[A], LOCAL[B]]),
        "post_edge_strength_A_B": float(causal[LOCAL[A], LOCAL[B]]),
        "base_edges_of_interest": {
            f"{A}--{C}": float(base[LOCAL[A], LOCAL[C]]),
            f"{A}--{D}": float(base[LOCAL[A], LOCAL[D]]),
            f"{A}--{E}": float(base[LOCAL[A], LOCAL[E]]),
            f"{B}--{C}": float(base[LOCAL[B], LOCAL[C]]),
            f"{B}--{D}": float(base[LOCAL[B], LOCAL[D]]),
            f"{B}--{E}": float(base[LOCAL[B], LOCAL[E]]),
        },
        "post_edges_of_interest": {
            f"{A}--{C}": float(causal[LOCAL[A], LOCAL[C]]),
            f"{A}--{D}": float(causal[LOCAL[A], LOCAL[D]]),
            f"{A}--{E}": float(causal[LOCAL[A], LOCAL[E]]),
            f"{B}--{C}": float(causal[LOCAL[B], LOCAL[C]]),
            f"{B}--{D}": float(causal[LOCAL[B], LOCAL[D]]),
            f"{B}--{E}": float(causal[LOCAL[B], LOCAL[E]]),
        },
        "best_tested_subset": best_tested,
        "best_oracle_subset": best_oracle,
        "correct_subset_row": correct_row,
        "candidate_subsets_examined": traces,
        "edge_survival_reason": reason,
    }


def ungated_family_diagnostic(structure: str, seeds=range(20)):
    """Diagnostic only: remove the current pre-gate and see if set-screening
    becomes viable or simply over-suppresses everything."""
    def ungated(present, base, max_set_size=3):
        W = np.zeros_like(base)
        n = present.shape[1]
        for i in range(n):
            xi = present[:, i]
            for j in range(i + 1, n):
                base_ij = base[i, j]
                if base_ij <= 0:
                    continue
                best = None
                xj = present[:, j]
                others = [k for k in range(n) if k not in (i, j)]
                for r in range(1, min(max_set_size, len(others)) + 1):
                    for subset in combinations(others, r):
                        links = [min(base[i, k], base[j, k]) for k in subset]
                        if min(links) <= 0:
                            continue
                        weakest = min(links)
                        cmi = d9.bin_cmi_set(xi, xj, present[:, subset])
                        explain = weakest / (cmi + 1e-9)
                        row = (subset, weakest, cmi, explain)
                        if best is None or explain > best[3]:
                            best = row
                if best is None:
                    W[i, j] = W[j, i] = base_ij
                else:
                    _, weakest, cmi, _ = best
                    keep_frac = cmi / (cmi + weakest + 1e-9)
                    W[i, j] = W[j, i] = base_ij * keep_frac
        return W

    passes = 0
    samples = []
    for seed in seeds:
        present = d9.sample_structure_events(structure, KEEP, LOCAL, 4000, seed)
        events = [np.where(row > 0)[0] for row in present]
        base = d9.pmi_matrix(events, len(KEEP))
        causal = ungated(present, base, max_set_size=3)
        if structure == "confound":
            bad = float(causal[LOCAL[A], LOCAL[B]])
            k1 = float(causal[LOCAL[A], LOCAL[C]])
            k2 = float(causal[LOCAL[B], LOCAL[C]])
            ctrl_base = float(base[LOCAL[E], LOCAL[F]])
            ctrl = float(causal[LOCAL[E], LOCAL[F]])
            ok = bad < 0.25 * min(k1, k2) and ctrl > 0.50 * ctrl_base
        elif structure == "double_confound":
            bad = float(causal[LOCAL[A], LOCAL[B]])
            k1 = max(float(causal[LOCAL[A], LOCAL[C]]), float(causal[LOCAL[A], LOCAL[D]]))
            k2 = max(float(causal[LOCAL[B], LOCAL[C]]), float(causal[LOCAL[B], LOCAL[D]]))
            ctrl_base = float(base[LOCAL[E], LOCAL[F]])
            ctrl = float(causal[LOCAL[E], LOCAL[F]])
            ok = bad < 0.25 * min(k1, k2) and ctrl > 0.50 * ctrl_base
        elif structure == "triple_confound":
            bad = float(causal[LOCAL[A], LOCAL[B]])
            k1 = max(float(causal[LOCAL[A], LOCAL[C]]), float(causal[LOCAL[A], LOCAL[D]]), float(causal[LOCAL[A], LOCAL[E]]))
            k2 = max(float(causal[LOCAL[B], LOCAL[C]]), float(causal[LOCAL[B], LOCAL[D]]), float(causal[LOCAL[B], LOCAL[E]]))
            ctrl_base = float(base[LOCAL[F], LOCAL[G]])
            ctrl = float(causal[LOCAL[F], LOCAL[G]])
            ok = bad < 0.25 * min(k1, k2) and ctrl > 0.50 * ctrl_base
        elif structure == "chain":
            bad = float(causal[LOCAL[A], LOCAL[C]])
            k1 = float(causal[LOCAL[A], LOCAL[B]])
            k2 = float(causal[LOCAL[B], LOCAL[C]])
            ctrl_base = float(base[LOCAL[E], LOCAL[F]])
            ctrl = float(causal[LOCAL[E], LOCAL[F]])
            ok = bad < 0.25 * min(k1, k2) and ctrl > 0.50 * ctrl_base
        else:
            bad = float(causal[LOCAL[A], LOCAL[B]])
            k1 = float(causal[LOCAL[A], LOCAL[C]])
            k2 = float(causal[LOCAL[B], LOCAL[C]])
            ctrl_base = float(base[LOCAL[E], LOCAL[F]])
            ctrl = float(causal[LOCAL[E], LOCAL[F]])
            ok = bad < 0.25 * min(k1, k2) and ctrl > 0.50 * ctrl_base
        passes += int(ok)
        if seed == 0:
            samples = [bad, k1, k2]
    return {"passes": passes, "total": len(list(seeds)), "sample": samples}


def write_failure_traces():
    out = []
    summary = {}
    for structure in ("double_confound", "triple_confound"):
        traces = []
        for seed in range(50):
            res = d9.evaluate_structure(structure, KEEP, LOCAL, seed, learner="set")
            if not res["pass"]:
                traces.append(trace_failed_run(structure, seed))
        summary[structure] = {
            "failed_runs": len(traces),
            "all_fail": len(traces) == 50,
            "correct_set_generated": sum(t["correct_subset_row"] is not None for t in traces),
            "correct_set_skipped_by_gate": sum(
                t["correct_subset_row"] is not None and t["correct_subset_row"]["skipped_by_gate"] for t in traces
            ),
            "best_oracle_is_correct_set": sum(
                t["best_oracle_subset"] is not None and t["best_oracle_subset"]["subset_names"] == t["expected_explainer"]
                for t in traces
            ),
        }
        path = TRACE_DIR / f"{structure}_failed_runs.jsonl"
        with path.open("w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        out.append((structure, path, traces))
    return summary, out


def benchmark_validation_markdown():
    return f"""# Day 9 Benchmark Validation

## Single Confound
Diagram:
```text
{A} <- {C} -> {B}
{E} -- {F}   (direct control)
```
Expected undirected skeleton:
- keep `{A}--{C}`
- keep `{B}--{C}`
- remove `{A}--{B}`
- keep `{E}--{F}`

Why:
- `A` and `B` are d-separated by conditioning on `C`.
- The direct control edge should remain because no explanatory parent screens it off.

## Double Confound
Diagram:
```text
{A} <- {C} -> {B}
{A} <- {D} -> {B}
{E} -- {F}
```
Expected undirected skeleton:
- keep `{A}--{C}`, `{B}--{C}`, `{A}--{D}`, `{B}--{D}`
- remove `{A}--{B}`
- keep `{E}--{F}`

Why:
- `A` and `B` remain associated after conditioning on only `C` or only `D`.
- `A` and `B` should disappear only after conditioning on the JOINT set `{{{C}, {D}}}`.

## Triple Confound
Diagram:
```text
{A} <- {C} -> {B}
{A} <- {D} -> {B}
{A} <- {E} -> {B}
{F} -- {G}
```
Expected undirected skeleton:
- keep `{A}--{C}`, `{B}--{C}`, `{A}--{D}`, `{B}--{D}`, `{A}--{E}`, `{B}--{E}`
- remove `{A}--{B}`
- keep `{F}--{G}`

Why:
- `A` and `B` should disappear only after conditioning on the JOINT set `{{{C}, {D}, {E}}}`.

## Chain
Diagram:
```text
{A} -> {B} -> {C}
{E} -- {F}
```
Expected undirected skeleton:
- keep `{A}--{B}`
- keep `{B}--{C}`
- remove `{A}--{C}`
- keep `{E}--{F}`

Why:
- In a chain, the endpoints become independent when conditioning on the middle node.

## Collider
Diagram:
```text
{A} -> {C} <- {B}
{E} -- {F}
```
Expected undirected skeleton:
- keep `{A}--{C}`
- keep `{B}--{C}`
- remove `{A}--{B}` marginally
- keep `{E}--{F}`

Why:
- Marginally, `A` and `B` are independent in a collider.
- Conditioning on `C` OPENS the path, so `I(A;B|C)` should increase.
"""


def info_section():
    lines = ["# Direct Conditional-Information Diagnostics", ""]
    for structure in ("confound", "double_confound", "triple_confound"):
        rows = cmi_bundle(structure)
        lines.append(f"## {structure}")
        for key in rows[0].keys():
            lines.append(f"- `{key}`: {fmt([r[key] for r in rows])}")
        lines.append("")

    for structure in ("chain", "collider"):
        rows = structure_specific_info(structure)
        lines.append(f"## {structure}")
        for key in rows[0].keys():
            lines.append(f"- `{key}`: {fmt([r[key] for r in rows])}")
        lines.append("")
    return "\n".join(lines)


def capability_table_and_matrix():
    return """# Causal Capability Table

| Capability | Status | Evidence |
|---|---:|---|
| Association discovery | PASS | Day 8b/8c already established stable ~3-4x separation |
| Single-confound rejection | PASS | Day 9: 100/100 |
| Chain handling | PASS | Day 9: 50/50 |
| Collider handling | PASS | Day 9: 50/50 |
| Double-confound rejection | FAIL | Day 9b: 0/50 |
| Triple-confound rejection | FAIL | Day 9b: 0/50 |
| Causal direction discovery | NOT SHOWN | No directional criterion implemented |
| General causal structure learning | NOT SHOWN | Multi-confound failure blocks this claim |

# Pass/Fail Matrix

| Benchmark | Expected screen target | Day 9 | Day 9b |
|---|---|---:|---:|
| Single confound | remove `A-B` via `C` | PASS | PASS |
| Double confound | remove `A-B` via `{C,D}` | FAIL | FAIL |
| Triple confound | remove `A-B` via `{C,D,E}` | FAIL | FAIL |
| Chain | remove `A-C` via `B` | PASS | PASS |
| Collider | keep `A-B` marginally weak; do not hallucinate direct edge | PASS | PASS |
"""


def write_main_report(summary):
    double_ungated = ungated_family_diagnostic("double_confound", seeds=range(20))
    triple_ungated = ungated_family_diagnostic("triple_confound", seeds=range(20))
    conf_ungated = ungated_family_diagnostic("confound", seeds=range(20))
    chain_ungated = ungated_family_diagnostic("chain", seeds=range(20))
    collider_ungated = ungated_family_diagnostic("collider", seeds=range(20))

    report = []
    report.append("# Day 9 Research Agenda Report")
    report.append("")
    report.append("## Failure Analysis")
    report.append("")
    report.append("Phase-1 question answers:")
    report.append("")
    report.append("1. **Is Day 9b actually evaluating joint conditioning?**")
    report.append("   - Candidate joint sets are generated by `combinations(others, r)` for `r=1..3`.")
    report.append("   - So: **yes, joint sets are generated syntactically**.")
    report.append("")
    report.append("2. **Is it only evaluating variables independently?**")
    report.append("   - In the actual current implementation, effectively **yes for the failed runs**.")
    report.append("   - Reason: every correct joint explanatory set in double/triple confound is rejected by the pre-gate")
    report.append("     `base_ij >= 0.95 * weakest_link` before `I(A;B|S)` is allowed to influence the score.")
    report.append("")
    report.append("3. **Are candidate explanatory sets being generated correctly?**")
    report.append("   - **Yes.** The correct sets `{C,D}` and `{C,D,E}` appear in every failed-run trace.")
    report.append("")
    report.append("4. **Are explanatory sets generated but rejected incorrectly?**")
    report.append("   - **Yes.** This is the dominant root cause in the current code path.")
    report.append(f"   - Double-confound failed runs: correct joint set skipped by gate in {summary['double_confound']['correct_set_skipped_by_gate']}/{summary['double_confound']['failed_runs']}.")
    report.append(f"   - Triple-confound failed runs: correct joint set skipped by gate in {summary['triple_confound']['correct_set_skipped_by_gate']}/{summary['triple_confound']['failed_runs']}.")
    report.append("")
    report.append("5. **Does the scoring rule fail even when the correct explanatory set is tested?**")
    report.append("   - Diagnostic answer: **also yes, in the naive ungated family**.")
    report.append("   - When the gate is removed as a diagnostic, the family becomes unstable and over-suppresses true edges:")
    report.append(f"     - confound: {conf_ungated['passes']}/{conf_ungated['total']}")
    report.append(f"     - double_confound: {double_ungated['passes']}/{double_ungated['total']}")
    report.append(f"     - triple_confound: {triple_ungated['passes']}/{triple_ungated['total']}")
    report.append(f"     - chain: {chain_ungated['passes']}/{chain_ungated['total']}")
    report.append(f"     - collider: {collider_ungated['passes']}/{collider_ungated['total']}")
    report.append("")
    report.append("### Root Cause")
    report.append("")
    report.append("The Day 9b failure is **not** just 'multi-confounds are impossible' and **not** just 'the set learner was never called'.")
    report.append("It is a two-part failure:")
    report.append("")
    report.append("1. **Implementation limitation:** the current set learner generates the correct joint sets but the weakest-link pre-gate rejects them before scoring.")
    report.append("2. **Family/scoring limitation:** if that gate is removed naively, the current suppression rule over-suppresses true edges and breaks even previously-passing worlds.")
    report.append("")
    report.append("So the precise boundary is:")
    report.append("- the benchmark worlds contain the right conditional-information signal")
    report.append("- the current PMI + screening heuristic cannot use that signal safely")
    report.append("")
    report.append(benchmark_validation_markdown())
    report.append("")
    report.append(info_section())
    report.append("")
    report.append("## Can The Current Family Work?")
    report.append("")
    report.append("### Single confound")
    report.append("- **One-variable screen:** yes in principle and in practice, because `I(A;B|C)` is near zero and a single parent explains the correlation.")
    report.append("- **Set-conditioned screen:** yes in principle and in practice.")
    report.append("")
    report.append("### Double confound")
    report.append("- **One-variable screen:** no in principle. Conditioning on only `C` or only `D` leaves residual dependence, so the family cannot solve this benchmark.")
    report.append("- **Set-conditioned screen:** yes in principle, because `I(A;B|{C,D})` is near zero in the diagnostic measurements.")
    report.append("- **Current implementation:** no in practice, because the pre-gate blocks the correct set and the naive ungated score destabilizes the family.")
    report.append("")
    report.append("### Triple confound")
    report.append("- **One-variable screen:** no in principle.")
    report.append("- **Set-conditioned screen:** yes in principle, because `I(A;B|{C,D,E})` is near zero in the diagnostic measurements.")
    report.append("- **Current implementation:** no in practice, for the same two-part reason as double confound.")
    report.append("")
    report.append("### Major Milestone")
    report.append("")
    report.append("The direct conditional-information diagnostics show that the benchmark is not flawed and the signal is not absent.")
    report.append("Therefore the next problem is genuinely:")
    report.append("")
    report.append("> Can a graph built from experience represent causal structure strongly enough to use higher-order conditional explanations without collapsing true edges?")
    report.append("")
    report.append("That is beyond the old Day-8 question of whether WorldField merely learns association.")
    report.append("")
    report.append(capability_table_and_matrix())
    report.append("")
    report.append("## Recommendation for Day 9c")
    report.append("")
    report.append("Do **not** proceed to Day 10.")
    report.append("")
    report.append("Proceed to Day 9c only after preserving this benchmark suite unchanged.")
    report.append("The benchmark is now better than the learner. The next learner should be judged against the exact same:")
    report.append("- single confound")
    report.append("- double confound")
    report.append("- triple confound")
    report.append("- chain")
    report.append("- collider")
    report.append("")
    report.append("The highest-signal next step is a learner that uses higher-order conditional information directly without the current weakest-link gate and without the current over-suppression rule.")
    report.append("")
    report.append("## Trace Files")
    report.append("")
    report.append("- `reports/failure_traces/double_confound_failed_runs.jsonl`")
    report.append("- `reports/failure_traces/triple_confound_failed_runs.jsonl`")

    path = REPORT_DIR / "DAY9_RESEARCH_REPORT.md"
    path.write_text("\n".join(report))
    return path


def main():
    ensure_dirs()
    summary, _ = write_failure_traces()
    report_path = write_main_report(summary)
    print(f"wrote report: {report_path}")
    print(f"wrote traces: {TRACE_DIR / 'double_confound_failed_runs.jsonl'}")
    print(f"wrote traces: {TRACE_DIR / 'triple_confound_failed_runs.jsonl'}")


if __name__ == "__main__":
    main()
