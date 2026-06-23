"""Day 9 — causal structure: can we suppress a confounded edge?

Smallest honest benchmark:
  - A and B co-occur often because of a third variable C.
  - The true direct relations are A--C and B--C.
  - A--B is ONLY correlation via C and should be suppressed.

This is intentionally concept-scale, not fragment-scale. Day 8c established that
granularity mismatch can fake fragility in graph learning; Day 9 should isolate
the CAUSAL question itself, not reopen the measurement-scale argument.

We compare:
  1. PMI baseline       : above-chance association only
  2. screened-PMI       : keep pairwise association, but suppress a pair if some
                          third variable largely screens it off (low I(X;Y|Z))

This does NOT learn arrow direction. It only asks for the undirected SKELETON:
  direct edges stay strong; explained-away correlations should weaken.
"""
import math
import os
import sys
from itertools import combinations

import numpy as np
import torch

DAY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_one"))
sys.path.insert(0, DAY1)
from config_rich import RichConfig  # noqa: E402
from data import ShapesDataset      # noqa: E402
from model import Worldfield        # noqa: E402

CKPT = os.path.join(DAY1, "out", "worldfield_rich.pt")


def device():
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")


def bin_mi(x, y):
    """Empirical mutual information for binary {0,1} arrays."""
    n = len(x)
    out = 0.0
    for xv in (0, 1):
        for yv in (0, 1):
            pxy = np.mean((x == xv) & (y == yv))
            if pxy <= 0:
                continue
            px = np.mean(x == xv)
            py = np.mean(y == yv)
            out += pxy * math.log((pxy / (px * py)) + 1e-12)
    return float(out)


def bin_cmi(x, y, z):
    """Empirical conditional mutual information I(X;Y|Z) for binary arrays."""
    out = 0.0
    for zv in (0, 1):
        mask = z == zv
        pz = float(mask.mean())
        if pz <= 0:
            continue
        out += pz * bin_mi(x[mask], y[mask])
    return float(out)


def bin_cmi_set(x, y, z_cols):
    """Empirical conditional mutual information I(X;Y|S) for binary set S."""
    if z_cols.ndim == 1:
        return bin_cmi(x, y, z_cols)
    out = 0.0
    n_cond = z_cols.shape[1]
    for mask_bits in range(1 << n_cond):
        mask = np.ones(len(x), dtype=bool)
        for j in range(n_cond):
            bit = (mask_bits >> j) & 1
            mask &= (z_cols[:, j] == bit)
        pz = float(mask.mean())
        if pz <= 0:
            continue
        out += pz * bin_mi(x[mask], y[mask])
    return float(out)


def pmi_counts(events, n_units):
    """Concept-scale PMI counts over event bags of unit ids."""
    ni = np.zeros(n_units, dtype=np.float64)
    nij = {}
    for ids in events:
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        ni[ids] += 1
        for i, j in combinations(ids.tolist(), 2):
            key = (min(i, j), max(i, j))
            nij[key] = nij.get(key, 0) + 1
    return ni, nij, len(events)


def pmi_matrix(events, n_units):
    ni, nij, n_events = pmi_counts(events, n_units)
    W = np.zeros((n_units, n_units), dtype=np.float64)
    n_events = max(n_events, 1)
    for (i, j), cij in nij.items():
        pij = cij / n_events
        pi = ni[i] / n_events
        pj = ni[j] / n_events
        if pi <= 0 or pj <= 0:
            continue
        pmi = math.log((pij / (pi * pj)) + 1e-12)
        if pmi > 0:
            W[i, j] = W[j, i] = pmi * cij
    return W


def screened_pmi(present, base):
    """Suppress pairwise edges that are screened off by a third variable.

    For each pair (i,j), find the third variable k that best "explains" the pair:
      - i and k strongly associated
      - j and k strongly associated
      - I(i;j|k) is small

    The stronger the explanation and the smaller the conditional dependence, the
    harder we suppress i--j. This is a tiny causal heuristic, not a full learner.
    """
    n = present.shape[1]
    W = np.zeros_like(base)
    cmi_cache = {}
    for i in range(n):
        xi = present[:, i]
        for j in range(i + 1, n):
            base_ij = base[i, j]
            if base_ij <= 0:
                continue
            best = 0.0
            best_k = None
            best_cmi = None
            xj = present[:, j]
            for k in range(n):
                if k in (i, j):
                    continue
                link = min(base[i, k], base[j, k])
                if link <= 0:
                    continue
                # Only suppress the WEAKEST edge in a triangle. If i--j is as
                # strong as one of the two links through k, treat it as a
                # plausible direct relation rather than an explained-away one.
                if base_ij >= 0.95 * link:
                    continue
                key = (i, j, k)
                if key not in cmi_cache:
                    cmi_cache[key] = bin_cmi(xi, xj, present[:, k])
                cmi = cmi_cache[key]
                # large when k explains i--j better than their residual dep.
                explain = link / (cmi + 1e-9)
                if explain > best:
                    best = explain
                    best_k = k
                    best_cmi = cmi
            if best_k is None:
                W[i, j] = W[j, i] = base_ij
                continue
            link = min(base[i, best_k], base[j, best_k])
            keep_frac = best_cmi / (best_cmi + link + 1e-9)
            W[i, j] = W[j, i] = base_ij * keep_frac
    return W


def screened_pmi_sets(present, base, max_set_size=3):
    """Suppress pairwise edges using conditional independence over SETS.

    Day 9b: identical benchmark, stronger learner. Search small explanatory
    sets S and suppress i--j if:
      - every node in S links strongly to both i and j
      - I(i;j|S) is small

    This still does not discover direction; it is skeleton cleaning with
    higher-order conditioning.
    """
    n = present.shape[1]
    W = np.zeros_like(base)
    cmi_cache = {}
    for i in range(n):
        xi = present[:, i]
        for j in range(i + 1, n):
            base_ij = base[i, j]
            if base_ij <= 0:
                continue
            best = 0.0
            best_set = None
            best_cmi = None
            xj = present[:, j]
            others = [k for k in range(n) if k not in (i, j)]
            for r in range(1, min(max_set_size, len(others)) + 1):
                for subset in combinations(others, r):
                    links = [min(base[i, k], base[j, k]) for k in subset]
                    if min(links) <= 0:
                        continue
                    weakest = min(links)
                    # Only suppress the weakest edge in an explained motif.
                    if base_ij >= 0.95 * weakest:
                        continue
                    key = (i, j, subset)
                    if key not in cmi_cache:
                        cmi_cache[key] = bin_cmi_set(xi, xj, present[:, subset])
                    cmi = cmi_cache[key]
                    explain = weakest / (cmi + 1e-9)
                    if explain > best:
                        best = explain
                        best_set = subset
                        best_cmi = cmi
            if best_set is None:
                W[i, j] = W[j, i] = base_ij
                continue
            weakest = min(min(base[i, k], base[j, k]) for k in best_set)
            keep_frac = best_cmi / (best_cmi + weakest + 1e-9)
            W[i, j] = W[j, i] = base_ij * keep_frac
    return W


def w(mat, local, x, y):
    return float(mat[local[x], local[y]])


def row(mat, keep, local, x):
    i = local[x]
    pairs = [(keep[j], mat[i, j]) for j in range(len(keep)) if j != i]
    return sorted(pairs, key=lambda t: -t[1])


def sample_structure_events(structure, keep, local, n_events, seed):
    """Binary event generators for several causal/statistical structures."""
    rng = np.random.RandomState(seed)
    present = np.zeros((n_events, len(keep)), dtype=np.int64)

    A, B, C, D, E, F, G = keep
    for t in range(n_events):
        act = np.zeros(len(keep), dtype=np.int64)

        if structure == "confound":
            # C -> A and C -> B ; A--B should be explained away by C
            c_on = rng.rand() < 0.35
            if c_on:
                act[local[C]] = 1
                if rng.rand() < 0.78:
                    act[local[A]] = 1
                if rng.rand() < 0.78:
                    act[local[B]] = 1
            else:
                if rng.rand() < 0.05:
                    act[local[A]] = 1
                if rng.rand() < 0.05:
                    act[local[B]] = 1

        elif structure == "double_confound":
            # C -> A,B and D -> A,B ; A--B should still be explained away.
            c_on = rng.rand() < 0.25
            d_on = rng.rand() < 0.25
            if c_on:
                act[local[C]] = 1
                if rng.rand() < 0.72:
                    act[local[A]] = 1
                if rng.rand() < 0.72:
                    act[local[B]] = 1
            if d_on:
                act[local[D]] = 1
                if rng.rand() < 0.72:
                    act[local[A]] = 1
                if rng.rand() < 0.72:
                    act[local[B]] = 1
            if not c_on and not d_on:
                if rng.rand() < 0.04:
                    act[local[A]] = 1
                if rng.rand() < 0.04:
                    act[local[B]] = 1

        elif structure == "triple_confound":
            # C -> A,B and D -> A,B and E -> A,B ; if this fails exactly like
            # double_confound, the limitation is "condition on one variable at a
            # time", not a quirky two-parent edge case.
            c_on = rng.rand() < 0.18
            d_on = rng.rand() < 0.18
            e_on = rng.rand() < 0.18
            if c_on:
                act[local[C]] = 1
                if rng.rand() < 0.72:
                    act[local[A]] = 1
                if rng.rand() < 0.72:
                    act[local[B]] = 1
            if d_on:
                act[local[D]] = 1
                if rng.rand() < 0.72:
                    act[local[A]] = 1
                if rng.rand() < 0.72:
                    act[local[B]] = 1
            if e_on:
                act[local[E]] = 1
                if rng.rand() < 0.72:
                    act[local[A]] = 1
                if rng.rand() < 0.72:
                    act[local[B]] = 1
            if not c_on and not d_on and not e_on:
                if rng.rand() < 0.04:
                    act[local[A]] = 1
                if rng.rand() < 0.04:
                    act[local[B]] = 1

        elif structure == "chain":
            # A -> B -> C ; A--C should be explained away by B.
            a_on = rng.rand() < 0.28
            if a_on:
                act[local[A]] = 1
                if rng.rand() < 0.86:
                    act[local[B]] = 1
                    if rng.rand() < 0.86:
                        act[local[C]] = 1
                elif rng.rand() < 0.04:
                    act[local[C]] = 1
            else:
                if rng.rand() < 0.03:
                    act[local[B]] = 1
                    if rng.rand() < 0.50:
                        act[local[C]] = 1

        elif structure == "collider":
            # A -> C <- B ; A--B should stay weak marginally and after screen.
            a_on = rng.rand() < 0.25
            b_on = rng.rand() < 0.25
            if a_on:
                act[local[A]] = 1
            if b_on:
                act[local[B]] = 1
            if (a_on and rng.rand() < 0.82) or (b_on and rng.rand() < 0.82):
                act[local[C]] = 1

        else:
            raise ValueError(structure)

        # one true direct control pair that should survive screening everywhere.
        # In triple_confound, E is already used as a confound, so move the direct
        # control one step over to F--G.
        if structure == "triple_confound":
            ctrl_a, ctrl_b = F, G
        else:
            ctrl_a, ctrl_b = E, F
        ctrl_on = rng.rand() < 0.22
        if ctrl_on:
            act[local[ctrl_a]] = 1
            if rng.rand() < 0.88:
                act[local[ctrl_b]] = 1
        elif rng.rand() < 0.04:
            act[local[ctrl_b]] = 1

        # mild distractor background
        if structure == "triple_confound":
            if rng.rand() < 0.04:
                act[local[C]] = act[local[C]] or 1
        elif rng.rand() < 0.06:
            act[local[G]] = 1

        present[t] = act
    return present


def evaluate_structure(structure, keep, local, seed, n_events=4000, learner="single"):
    present = sample_structure_events(structure, keep, local, n_events, seed)
    events = [np.where(row > 0)[0] for row in present]
    base = pmi_matrix(events, len(keep))
    if learner == "single":
        causal = screened_pmi(present, base)
    elif learner == "set":
        causal = screened_pmi_sets(present, base, max_set_size=3)
    else:
        raise ValueError(learner)

    A, B, C, D, E, F, G = keep
    metrics = {
        "seed": seed,
        "structure": structure,
        "learner": learner,
        "base": base,
        "causal": causal,
        "present": present,
    }

    if structure == "confound":
        metrics.update({
            "target_bad": w(causal, local, A, B),
            "target_bad_pmi": w(base, local, A, B),
            "keeper_1": w(causal, local, A, C),
            "keeper_2": w(causal, local, B, C),
            "keeper_1_pmi": w(base, local, A, C),
            "keeper_2_pmi": w(base, local, B, C),
            "explain": bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[C]]),
            "kind": "confound",
        })
    elif structure == "double_confound":
        metrics.update({
            "target_bad": w(causal, local, A, B),
            "target_bad_pmi": w(base, local, A, B),
            "keeper_1": max(w(causal, local, A, C), w(causal, local, A, D)),
            "keeper_2": max(w(causal, local, B, C), w(causal, local, B, D)),
            "keeper_1_pmi": max(w(base, local, A, C), w(base, local, A, D)),
            "keeper_2_pmi": max(w(base, local, B, C), w(base, local, B, D)),
            "explain": min(
                bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[C]]),
                bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[D]]),
            ),
            "kind": "confound",
        })
    elif structure == "triple_confound":
        metrics.update({
            "target_bad": w(causal, local, A, B),
            "target_bad_pmi": w(base, local, A, B),
            "keeper_1": max(w(causal, local, A, C), w(causal, local, A, D), w(causal, local, A, E)),
            "keeper_2": max(w(causal, local, B, C), w(causal, local, B, D), w(causal, local, B, E)),
            "keeper_1_pmi": max(w(base, local, A, C), w(base, local, A, D), w(base, local, A, E)),
            "keeper_2_pmi": max(w(base, local, B, C), w(base, local, B, D), w(base, local, B, E)),
            "explain": min(
                bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[C]]),
                bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[D]]),
                bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[E]]),
            ),
            "kind": "confound",
        })
    elif structure == "chain":
        metrics.update({
            "target_bad": w(causal, local, A, C),
            "target_bad_pmi": w(base, local, A, C),
            "keeper_1": w(causal, local, A, B),
            "keeper_2": w(causal, local, B, C),
            "keeper_1_pmi": w(base, local, A, B),
            "keeper_2_pmi": w(base, local, B, C),
            "explain": bin_cmi(present[:, local[A]], present[:, local[C]], present[:, local[B]]),
            "kind": "chain",
        })
    elif structure == "collider":
        metrics.update({
            "target_bad": w(causal, local, A, B),
            "target_bad_pmi": w(base, local, A, B),
            "keeper_1": w(causal, local, A, C),
            "keeper_2": w(causal, local, B, C),
            "keeper_1_pmi": w(base, local, A, C),
            "keeper_2_pmi": w(base, local, B, C),
            "explain": bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[C]]),
            "kind": "collider",
        })

    if structure == "triple_confound":
        metrics["control_pmi"] = w(base, local, F, G)
        metrics["control"] = w(causal, local, F, G)
        metrics["distractor"] = max(w(causal, local, A, C), w(causal, local, B, D))
    else:
        metrics["control_pmi"] = w(base, local, E, F)
        metrics["control"] = w(causal, local, E, F)
        metrics["distractor"] = max(w(causal, local, A, G), w(causal, local, B, G))

    if structure in ("confound", "double_confound", "triple_confound", "chain"):
        metrics["pass"] = (
            metrics["target_bad"] < 0.25 * min(metrics["keeper_1"], metrics["keeper_2"])
            and metrics["control"] > 0.50 * metrics["control_pmi"]
        )
    else:
        # collider: the screen should NOT hallucinate an A--B suppression story;
        # A--B should already be weak under PMI and stay weak.
        metrics["pass"] = (
            metrics["target_bad_pmi"] < 0.25 * min(metrics["keeper_1_pmi"], metrics["keeper_2_pmi"])
            and metrics["target_bad"] < 0.25 * min(metrics["keeper_1"], metrics["keeper_2"])
            and metrics["control"] > 0.50 * metrics["control_pmi"]
        )
    return metrics


def main():
    dev = device()
    ck = torch.load(CKPT, map_location=dev)
    cfg = RichConfig()
    model = Worldfield(cfg, ck["vocab_size"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    names = ck["class_names"]
    _ = ShapesDataset(cfg, "train")  # keep dependency shape consistent with the repo
    idx = {n: i for i, n in enumerate(names)}

    # roles for the benchmark. We reuse real learned concept ids, but the Day 9
    # test is concept-scale on purpose so it isolates the causal question.
    A = "magenta diamond"
    B = "skyblue pentagon"
    C = "crimson square"
    D = "red circle"
    E = "blue square"
    F = "teal pentagon"
    G = "olive square"

    keep = [A, B, C, D, E, F, G]
    local = {name: i for i, name in enumerate(keep)}
    one = evaluate_structure("confound", keep, local, seed=0, learner="single")
    base = one["base"]
    causal = one["causal"]
    present = one["present"]

    print("=== Day 9: confound rejection beyond association ===")
    print("scope:")
    print("  This is NOT direction learning.")
    print("  It is a concept-level conditional-independence screen over a PMI graph.\n")

    print("single benchmark (seed 0):")
    print(f"  confounded trio : {A} -- {C} -- {B}   (A--B is NOT direct)")
    print(f"  direct control  : {E} -- {F}")
    print(f"  distractor      : {G}\n")

    print("event frequencies:")
    for name in keep:
        print(f"  P({name:16s}) = {present[:, local[name]].mean():.3f}")

    print("\n=== 1. PMI baseline (association only) ===")
    print(f"  A--C: {w(base, local, A, C):8.2f}")
    print(f"  B--C: {w(base, local, B, C):8.2f}")
    print(f"  A--B: {w(base, local, A, B):8.2f}   <- confounded edge")
    print(f"  E--F: {w(base, local, E, F):8.2f}   <- direct control")
    print(f"  A top neighbors by PMI: {row(base, keep, local, A)[:4]}")

    print("\n=== 2. Screened-PMI (conditional-independence screen) ===")
    print(f"  A--C: {w(causal, local, A, C):8.2f}")
    print(f"  B--C: {w(causal, local, B, C):8.2f}")
    print(f"  A--B: {w(causal, local, A, B):8.2f}   <- should collapse if C screens it off")
    print(f"  E--F: {w(causal, local, E, F):8.2f}   <- should stay strong")
    print(f"  A top neighbors after screen: {row(causal, keep, local, A)[:4]}")

    print("\n=== 3. Why the screen acts ===")
    print(f"  I(A;B | C) = {bin_cmi(present[:, local[A]], present[:, local[B]], present[:, local[C]]):.4f}")
    print(f"  I(E;F | C) = {bin_cmi(present[:, local[E]], present[:, local[F]], present[:, local[C]]):.4f}")

    print("\n=== 4. Robustness across seeds ===")
    seeds = range(100)
    conf = [evaluate_structure("confound", keep, local, seed=s, learner="single") for s in seeds]
    n_ab = sum(r["target_bad"] < 0.25 * min(r["keeper_1"], r["keeper_2"]) for r in conf)
    n_ac = sum(r["keeper_1"] > 0.50 * r["keeper_1_pmi"] for r in conf)
    n_bc = sum(r["keeper_2"] > 0.50 * r["keeper_2_pmi"] for r in conf)
    n_ctrl = sum(r["control"] > 0.50 * r["control_pmi"] for r in conf)
    print(f"  A--B removed : {n_ab}/100")
    print(f"  A--C retained: {n_ac}/100")
    print(f"  B--C retained: {n_bc}/100")
    print(f"  E--F retained: {n_ctrl}/100")

    print("\n=== 5. Topology checks ===")
    topo_names = {
        "double_confound": "A<-C->B plus A<-D->B",
        "triple_confound": "A<-C->B plus A<-D->B plus A<-E->B",
        "chain": "A->B->C (screen A--C via B)",
        "collider": "A->C<-B (A--B should already stay weak)",
    }
    topo_results = {}
    for key, label in topo_names.items():
        runs = [evaluate_structure(key, keep, local, seed=s, learner="single") for s in range(50)]
        passes = sum(r["pass"] for r in runs)
        topo_results[key] = passes
        sample = runs[0]
        print(f"  {label:28s}: {passes}/50 pass | "
              f"bad={sample['target_bad']:.2f} keepers={sample['keeper_1']:.2f}/{sample['keeper_2']:.2f}")

    print("\n=== VERDICT (Day 9) ===")
    robust_confound = n_ab >= 95 and n_ac >= 95 and n_bc >= 95 and n_ctrl >= 95
    topo_ok = all(v >= 45 for v in topo_results.values())
    print(f"  confound rejection robust across seeds : {robust_confound}")
    print(f"  survives double-confound / chain / collider : {topo_ok}")
    if robust_confound and topo_ok:
        print("\nPASS — on these controlled concept-level benchmarks, the graph goes")
        print("beyond pure association: PMI wires confounded correlations, while a")
        print("conditional-independence screen suppresses them and preserves direct")
        print("edges across seeds and multiple causal topologies.")
        print("\nCareful claim: this is confound rejection / skeleton cleaning, not")
        print("causal direction discovery.")
    elif robust_confound:
        print("\nPARTIAL — the basic confound benchmark is robust, but at least one")
        print("additional topology still breaks the screen. This is beyond Day 8,")
        print("but not yet a general causal-structure result.")
    else:
        print("\nWEAK — the single benchmark worked, but robustness does not hold.")
        print("The result remains prototype-level rather than a stable Day 9 pass.")

    print("\n=== Day 9b: set-conditioning on the SAME harness ===")
    conf_b = [evaluate_structure("confound", keep, local, seed=s, learner="set") for s in seeds]
    n_ab_b = sum(r["target_bad"] < 0.25 * min(r["keeper_1"], r["keeper_2"]) for r in conf_b)
    n_ac_b = sum(r["keeper_1"] > 0.50 * r["keeper_1_pmi"] for r in conf_b)
    n_bc_b = sum(r["keeper_2"] > 0.50 * r["keeper_2_pmi"] for r in conf_b)
    n_ctrl_b = sum(r["control"] > 0.50 * r["control_pmi"] for r in conf_b)
    print(f"  single confound: A--B removed {n_ab_b}/100 | A--C retained {n_ac_b}/100 | "
          f"B--C retained {n_bc_b}/100 | control retained {n_ctrl_b}/100")

    topo_results_b = {}
    for key, label in topo_names.items():
        runs = [evaluate_structure(key, keep, local, seed=s, learner="set") for s in range(50)]
        passes = sum(r["pass"] for r in runs)
        topo_results_b[key] = passes
        sample = runs[0]
        print(f"  {label:28s}: {passes}/50 pass | "
              f"bad={sample['target_bad']:.2f} keepers={sample['keeper_1']:.2f}/{sample['keeper_2']:.2f}")

    robust_b = n_ab_b >= 95 and n_ac_b >= 95 and n_bc_b >= 95 and n_ctrl_b >= 95
    topo_ok_b = all(v >= 45 for v in topo_results_b.values())
    print("\n=== VERDICT (Day 9b) ===")
    print(f"  set-conditioning robust across seeds : {robust_b}")
    print(f"  set-conditioning survives full topology suite : {topo_ok_b}")
    if robust_b and topo_ok_b:
        print("\nPASS — conditioning on SETS removes the Day 9 failure mode. The same")
        print("benchmark suite that broke one-variable screening now passes, so the")
        print("limitation really was higher-order confounds rather than a toy artifact.")
    elif robust_b:
        print("\nPARTIAL — set-conditioning fixes some multi-confound failures, but the")
        print("full suite still exposes cases it cannot explain away.")
    else:
        print("\nFAIL — moving from one-variable to set-conditioning did not reliably")
        print("clear the Day 9 boundary. More than higher-order conditioning is needed.")


if __name__ == "__main__":
    main()
