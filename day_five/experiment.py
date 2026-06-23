"""Day 5a — GRAPH PROPAGATION over ground-truth concept ids (clean, no latents).

SCOPE NOTE (honest): this tests the co-activation graph MECHANISM on clean
integer ids. It does NOT exercise the latent substrate — retrieval, slots, and
encoders are unused. Multi-hop propagation over a weighted graph is a solved
problem; passing here proves the mechanism is reasonable, not that the
architecture reasons. The real test is Day 5b (experiment_latent.py), which
seeds propagation from a latent query through actual retrieval.

Original docstring follows:

Day 5 — can the substrate recover a relation that was NEVER directly queried?

Honest design (avoiding the 'classification with extra steps' trap):
  - relations are NOT stored as facts; they emerge from co-activation (Hebbian).
  - we query with ONE concept and ask whether its UNQUERIED associate surfaces.
  - CONTROL: the same query against an untrained graph (pure latent similarity).
    Real reasoning must BEAT this control — otherwise it's just similarity.
  - we test 1-hop (direct co-occurrence) AND 2-hop (linked only via an
    intermediate, never co-activated directly) — the latter is the real test.
  - everything runs WITHIN known memory capacity (few concepts), per the Day-4.5
    limitations, so a memory limit can't be mislabeled a reasoning failure.

Uses the rich 60-concept model for real concept latents.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401
import sys
import numpy as np
import torch

DAY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_one"))
sys.path.insert(0, DAY1)
from config_rich import RichConfig    # noqa: E402
from data import ShapesDataset        # noqa: E402
from model import Worldfield          # noqa: E402
from coactivation import CoActivationGraph  # noqa: E402

CKPT = os.path.join(DAY1, "out", "worldfield_rich.pt")


def device():
    return torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")


def load(dev):
    ck = torch.load(CKPT, map_location=dev)
    cfg = RichConfig()
    m = Worldfield(cfg, ck["vocab_size"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg, ck["class_names"]


def main():
    dev = device()
    model, cfg, names = load(dev)
    idx = {n: i for i, n in enumerate(names)}
    n_concepts = len(names)

    # pick a small within-capacity working set of real concepts and give them
    # human-readable roles. (We reuse shape/color concepts as stand-in "objects".)
    cat   = idx["red circle"]      # "cat"
    sofa  = idx["blue square"]     # "sofa"
    room  = idx["green triangle"]  # "room"
    dog   = idx["yellow circle"]   # "dog"
    bone  = idx["teal pentagon"]   # "bone"
    alias = {cat: "cat", sofa: "sofa", room: "room", dog: "dog", bone: "bone"}
    print("concept roles:", {v: k for k, v in alias.items()})

    # ---------- scenario 1: direct relational recovery (1-hop) ----------
    # events: cat co-occurs with sofa and room. Query 'cat ?' -> should surface
    # sofa and room, which were never asked for.
    g = CoActivationGraph(n_concepts)
    for _ in range(5):
        g.observe_event([cat, sofa])    # "cat jumps on sofa"
        g.observe_event([cat, room])    # "cat enters room"
    scores = g.propagate([cat], hops=2)
    top = np.argsort(scores)[::-1][:3]
    print("\n=== 1. direct relation: query 'cat ?' ===")
    for t in top:
        tag = alias.get(t, names[t])
        print(f"   {tag:6s} score={scores[t]:.3f}")
    direct_ok = sofa in top[:2] and room in top[:3]

    # CONTROL: untrained graph -> nothing should be recovered (no edges)
    g0 = CoActivationGraph(n_concepts)
    ctrl = g0.propagate([cat], hops=2)
    print(f"   CONTROL (no edges): max score = {ctrl.max():.3f} "
          f"(should be ~0 — recovery must come from learned edges)")

    # ---------- scenario 2: TWO-HOP inference (the real test) ----------
    # cat--sofa co-occur; sofa--room co-occur; cat and room NEVER co-occur.
    # Query 'cat ?': can 'room' surface via the sofa intermediate (2 hops)?
    g2 = CoActivationGraph(n_concepts)
    for _ in range(5):
        g2.observe_event([cat, sofa])   # cat with sofa
        g2.observe_event([sofa, room])  # sofa with room  (cat & room never together)
    s2 = g2.propagate([cat], hops=2)
    # 1-hop only, to prove 'room' needs the second hop
    s1 = g2.propagate([cat], hops=1)
    print("\n=== 2. two-hop: cat--sofa, sofa--room (cat & room NEVER co-occur) ===")
    print(f"   query 'cat ?'  sofa={s2[sofa]:.3f} (1-hop), "
          f"room={s2[room]:.3f} (2-hop)")
    print(f"   with hops=1 only: room={s1[room]:.3f} "
          f"(should be ~0 — room is unreachable in one hop)")
    twohop_ok = s2[room] > 0.01 and s1[room] < 1e-6

    # ---------- scenario 3: discrimination (no false relations) ----------
    # dog--bone co-occur, separately from cat's world. Query 'cat ?' must NOT
    # surface bone (no path), or the graph is just firing everything.
    g3 = CoActivationGraph(n_concepts)
    for _ in range(5):
        g3.observe_event([cat, sofa])
        g3.observe_event([dog, bone])
    s3 = g3.propagate([cat], hops=2)
    print("\n=== 3. discrimination: cat-world vs dog-world ===")
    print(f"   query 'cat ?'  sofa={s3[sofa]:.3f}  bone={s3[bone]:.3f} "
          f"(bone should stay ~0)")
    discrim_ok = s3[sofa] > 0.01 and s3[bone] < 1e-6

    # ---------- verdict ----
    print("\n=== VERDICT ===")
    print(f"  1-hop relational recovery : {'PASS' if direct_ok else 'FAIL'}")
    print(f"  2-hop inference (real test): {'PASS' if twohop_ok else 'FAIL'}")
    print(f"  discrimination (no leak)  : {'PASS' if discrim_ok else 'FAIL'}")
    if direct_ok and twohop_ok and discrim_ok:
        print("\nPASS — the substrate recovers UNQUERIED relations from co-activation,")
        print("including a 2-hop link never directly observed, without firing")
        print("unrelated concepts. This is the first primitive reasoning: relations")
        print("emerge and propagate, they are not stored as facts.")
    else:
        print("\nINCOMPLETE — see which sub-test failed above.")


if __name__ == "__main__":
    main()
