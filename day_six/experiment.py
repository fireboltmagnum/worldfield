"""Day 6 — iterative refinement: can looping increase signal-to-noise, and is it
GENUINE error-correction or just winner-take-all attraction?

Day 5c found the gap: single-shot propagation is inert to seed noise — it breaks
at >=40% contamination and more hops don't help. The Worldfield idea was always
a dynamical system ('reason while updating state'), not one pass. Day 6 makes
propagation a LOOP:

    seed -> propagate -> re-seed from result -> propagate -> ... -> converge

THE CENTRAL QUESTION (per the good-vs-fake warning):
  GOOD:  the loop uses graph STRUCTURE to clean noise, and can even OVERTURN a
         wrong initial plurality (dog ahead -> cat wins). Real error-correction.
  FAKE:  the loop just reinforces whatever is already strongest (winner-take-all
         attraction). Sharpens the leader; can NEVER flip a wrong leader.

The decisive discriminator: seed with the WRONG concept in the lead. If iteration
flips it to the correct answer, that's genuine correction — WTA cannot do it.
We also run an explicit WTA baseline so the fake signature is visible for compare.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401
import sys
import numpy as np
import torch

DAY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_one"))
DAY2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_two"))
DAY5 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_five"))
sys.path[:0] = [DAY1, DAY2, DAY5]
from config_rich import RichConfig    # noqa: E402
from data import ShapesDataset        # noqa: E402
from model import Worldfield          # noqa: E402
from store import FragmentStore       # noqa: E402
from experiment_latent import FragmentGraph  # noqa: E402

CKPT = os.path.join(DAY1, "out", "worldfield_rich.pt")
OUT = os.path.join(os.path.dirname(__file__), "out")


def device():
    return torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def main():
    dev = device()
    ck = torch.load(CKPT, map_location=dev)
    cfg = RichConfig()
    model = Worldfield(cfg, ck["vocab_size"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    names = ck["class_names"]; idx = {n: i for i, n in enumerate(names)}
    ds = ShapesDataset(cfg, "train")

    store = FragmentStore(cfg.latent_dim, use_hnsw=False)
    V, L = [], []
    for i in range(len(ds)):
        img, _, lab = ds[i]
        with torch.no_grad():
            V.append(model.encode_image(img.unsqueeze(0).to(dev)).cpu().numpy()[0])
        L.append(lab)
    store.add(np.array(V, np.float32), np.array(L, np.int64))
    nF = len(store)
    os.makedirs(OUT, exist_ok=True)

    @torch.no_grad()
    def text_q(c):
        for _, txt, lab in (ds[i] for i in range(len(ds))):
            if lab == idx[c]:
                return norm(model.encode_text(txt.unsqueeze(0).to(dev)).cpu().numpy()[0])

    def frags_of(concept, k):
        s, i, _ = store.search(text_q(concept)[None].astype(np.float32), k)
        return i[0], s[0]

    cat, sofa, room = "red circle", "blue square", "green triangle"
    dog, bone = "yellow circle", "teal pentagon"

    # learn the world (clean events): cat~{sofa,room}, dog~{bone}
    g = FragmentGraph(nF)
    for _ in range(5):
        for (a, b) in [(cat, sofa), (cat, room), (dog, bone)]:
            ia, sa = frags_of(a, 6); ib, sb = frags_of(b, 6)
            g.observe(np.concatenate([ia, ib]), weights=np.concatenate([sa, sb]))

    def concept_scores(frag_scores):
        per = {}
        for f in np.argsort(frag_scores)[::-1][:80]:
            if frag_scores[f] <= 0:
                break
            c = int(store.labels[f]); per[c] = per.get(c, 0.0) + frag_scores[f]
        return per

    def seed_from_concepts(concept_weights, k=6):
        """Build a fragment seed from {concept: weight}."""
        ids, vals = [], []
        for c_name, w in concept_weights.items():
            ci, cs = frags_of(c_name, k)
            ids.append(ci); vals.append(cs * w)
        return np.concatenate(ids), np.concatenate(vals).astype(np.float32)

    cat_i, dog_i, sofa_i, room_i, bone_i = (idx[cat], idx[dog], idx[sofa],
                                            idx[room], idx[bone])

    # which fragments actually carry edges — re-seeding must stay on these, since
    # the graph only knows fragments that co-fired during events, not 'concepts'.
    Wcsr = g.W.tocsr()
    connected = np.asarray((Wcsr.sum(axis=1)).ravel() > 0).ravel()
    print(f"(graph is sparse: {int(connected.sum())} of {nF} fragments have edges)")

    def refine_loop(init_concepts, iters=6, mode="refine", k_seed=6, hops=2,
                    keep_frac=0.5):
        """Iterate at the FRAGMENT level (not concept level): the graph only knows
        fragments that co-fired, so we carry fragment activation forward and
        re-seed from the propagated activation itself, restricted to CONNECTED
        fragments. (Re-seeding with fresh concept fragments hits unconnected nodes
        and dies — a real constraint the substrate imposes.)

        mode='refine' : re-seed from the full propagated fragment activation
                        (graph structure decides what stays lit)
        mode='wta'    : re-seed only from the single strongest fragment (the fake
                        winner-take-all baseline)
        """
        seed_ids, seed_vals = seed_from_concepts(init_concepts, k_seed)
        traj = []
        # carry a DENSE fragment-state vector so we can damp (EMA) between steps.
        # Undamped full-replacement re-seeding oscillates (it overshoots between a
        # concept and its associates) — damping is what turns the loop into a
        # convergent dynamical system. (The recurring lesson: state needs inertia.)
        state = np.zeros(nF, dtype=np.float32)
        state[seed_ids] = seed_vals / (seed_vals.max() + 1e-9)
        damp = 0.6   # fraction of previous state retained
        for it in range(iters):
            fs = g.propagate(np.where(state > 0)[0], state[state > 0], hops=hops)
            cs = concept_scores(fs)
            cat_sig = cs.get(sofa_i, 0) + cs.get(room_i, 0)
            dog_sig = cs.get(bone_i, 0)
            tot = cat_sig + dog_sig + 1e-9
            traj.append((cat_sig / tot, dog_sig / tot))

            lit = np.where(fs > 0)[0]
            if lit.size == 0:
                break
            new = np.zeros(nF, dtype=np.float32)
            if mode == "wta":
                top = lit[np.argsort(fs[lit])[::-1][:1]]
                new[top] = 1.0
            else:
                order = lit[np.argsort(fs[lit])[::-1]]
                keep = order[:max(1, int(len(order) * keep_frac))]
                new[keep] = fs[keep] / (fs[keep].max() + 1e-9)
            # EMA damping: blend new evidence with retained prior state
            state = damp * state + (1 - damp) * new
            state[state < 1e-4] = 0.0   # keep it sparse
        return traj

    def final_answer(init_concepts, mode, **kw):
        seed_ids, seed_vals = seed_from_concepts(init_concepts)
        # run loop, then read the final recovered associate ranking
        traj = refine_loop(init_concepts, mode=mode, **kw)
        # final propagation from the converged seed
        return traj

    def show_traj(title, traj):
        print(f"  {title}")
        for t, (c, d) in enumerate(traj):
            bar = "#" * int(c * 30)
            print(f"    iter {t}: cat-signal {c:5.2f}  dog-signal {d:5.2f}  {bar}")

    # ---------------- TEST 1: clean seed (sanity) ----------------
    print("=== TEST 1: clean seed (cat only) — should stay/strengthen cat ===")
    show_traj("refine", refine_loop({cat: 1.0}, mode="refine"))

    # ---------------- TEST 2: contaminated, cat still ahead -------
    print("\n=== TEST 2: cat 60% / dog 40% (the Day-5c level that broke single-shot) ===")
    show_traj("refine", refine_loop({cat: 0.6, dog: 0.4}, mode="refine"))

    # ---------------- TEST 3: THE DECISIVE TEST — dog AHEAD -------
    print("\n=== TEST 3 (decisive): dog 55% / cat 45% — WRONG concept leads ===")
    print("  GOOD architecture can OVERTURN this. Winner-take-all CANNOT.")
    refine_traj = refine_loop({cat: 0.45, dog: 0.55}, mode="refine")
    wta_traj = refine_loop({cat: 0.45, dog: 0.55}, mode="wta")
    show_traj("refine (uses graph structure)", refine_traj)
    show_traj("WTA baseline (fake-version signature)", wta_traj)

    # ---------------- verdict ----------------
    def converged(traj, tol=0.1):
        """Last 3 iterations must be stable (not oscillating)."""
        tail = [c for c, _ in traj[-3:]]
        return max(tail) - min(tail) < tol

    def flipped(traj):
        # cat starts behind, ENDS ahead, AND the ending is stable (converged) —
        # a lucky non-converged final value does NOT count.
        return traj[0][0] < 0.5 and traj[-1][0] > 0.5 and converged(traj)

    refine_flip = flipped(refine_traj)
    wta_flip = flipped(wta_traj)
    print(f"\n  [stability] refine converged: {converged(refine_traj)} | "
          f"WTA converged: {converged(wta_traj)}")
    # gain: how much did refine improve cat-signal vs WTA, from same start?
    refine_gain = refine_traj[-1][0] - refine_traj[0][0]
    wta_gain = wta_traj[-1][0] - wta_traj[0][0]

    print("\n=== VERDICT (Day 6) ===")
    print(f"  refine: cat-signal {refine_traj[0][0]:.2f} -> {refine_traj[-1][0]:.2f} "
          f"(overturned wrong lead: {'YES' if refine_flip else 'no'})")
    print(f"  WTA   : cat-signal {wta_traj[0][0]:.2f} -> {wta_traj[-1][0]:.2f} "
          f"(overturned: {'YES' if wta_flip else 'no'})")
    if refine_flip and not wta_flip:
        print("\nPASS (GOOD version) — iterative refinement OVERTURNS a wrong initial")
        print("plurality using graph structure; the WTA baseline cannot. This is")
        print("genuine error-correction, not attraction dynamics. The missing")
        print("capability (Day 5c) is supplied by iteration.")
    elif refine_flip and wta_flip:
        print("\nAMBIGUOUS — both flipped; the flip may be a propagation artifact, not")
        print("structure. Needs a harder discriminator.")
    elif not refine_flip and refine_gain > wta_gain + 0.05:
        print("\nPARTIAL — refine improves SNR more than WTA but cannot overturn a")
        print("wrong lead. Some structure used, but not full error-correction.")
    else:
        print("\nFAKE version / FAIL — refine behaves like winner-take-all (sharpens")
        print("the existing leader, cannot flip a wrong one). This is attraction")
        print("dynamics, not reasoning correction. The gap from Day 5c remains.")


if __name__ == "__main__":
    main()
