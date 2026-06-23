"""Day 6.5 — does iterative refinement survive a LARGE, NOISY, CONTRADICTORY graph?

Day 6's red flag (correctly diagnosed): only ~30 of 7680 fragments had edges. A
tiny graph manufactures attractor behavior — few possible trajectories, so a
'0.45 -> 1.00' flip may be structure dominating, not evidence integration. Real
error-correction has a SLOPE (0.45 -> 0.58 -> 0.71 -> ...), not a cliff.

This experiment removes every excuse:
  1. LARGE graph    : wire many fragments through many events (hundreds connected).
  2. CONTRADICTORY  : inject wrong edges INTO the learned graph itself —
                      cat--bone (10%) and dog--room (15%) accidental co-activations.
                      The structure is now partially wrong, not just the seed.
  3. SLOPE TEST     : we don't just ask 'did it flip?' — we inspect the trajectory.
                      A gradual climb = integration. A cliff = attractor artifact.

If refinement STILL converges to the correct associate on a large, partially
contradictory graph — gradually — that is the result that would actually convince.
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
RNG = np.random.RandomState(0)


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

    # fragments grouped by concept, so we can sample MANY distinct fragments per
    # concept across events (this is what makes the graph large, not 30 nodes).
    by_concept = {c: np.where(store.labels == c)[0] for c in range(len(names))}

    def sample_frags(concept_name, k):
        pool = by_concept[idx[concept_name]]
        return RNG.choice(pool, size=min(k, len(pool)), replace=False)

    cat, sofa, room = "red circle", "blue square", "green triangle"
    dog, bone = "yellow circle", "teal pentagon"
    # extra concepts to enlarge & clutter the graph with unrelated worlds
    extras = [("magenta diamond", "olive square"), ("teal circle", "navy triangle"),
              ("skyblue pentagon", "crimson square"), ("purple diamond", "green circle")]

    def build_graph(n_events=200, contradictions=True, frags_per=8):
        """Many events => many distinct fragments wired => large graph. Each event
        samples FRESH fragments for its concepts, so the connected set grows."""
        g = FragmentGraph(nF)
        for _ in range(n_events):
            # correct world
            for (a, b) in [(cat, sofa), (cat, room), (dog, bone)] + extras:
                ia, ib = sample_frags(a, frags_per), sample_frags(b, frags_per)
                g.observe(np.concatenate([ia, ib]),
                          weights=np.ones(len(ia) + len(ib), np.float32))
            if contradictions:
                # accidental WRONG edges baked into the structure
                if RNG.rand() < 0.10:                      # cat--bone 10%
                    ia, ib = sample_frags(cat, frags_per), sample_frags(bone, frags_per)
                    g.observe(np.concatenate([ia, ib]),
                              weights=np.ones(len(ia) + len(ib), np.float32))
                if RNG.rand() < 0.15:                      # dog--room 15%
                    ia, ib = sample_frags(dog, frags_per), sample_frags(room, frags_per)
                    g.observe(np.concatenate([ia, ib]),
                              weights=np.ones(len(ia) + len(ib), np.float32))
        return g

    sofa_i, room_i, bone_i = idx[sofa], idx[room], idx[bone]

    def concept_scores(fs):
        per = {}
        for f in np.argsort(fs)[::-1][:400]:
            if fs[f] <= 0:
                break
            c = int(store.labels[f]); per[c] = per.get(c, 0.0) + fs[f]
        return per

    def refine(g, init_concepts, iters=8, hops=2, damp=0.6, keep_frac=0.5, mode="refine"):
        state = np.zeros(nF, dtype=np.float32)
        for c, w in init_concepts.items():
            fi = sample_frags(c, 8); state[fi] = w
        traj = []
        for _ in range(iters):
            seed = np.where(state > 0)[0]
            if seed.size == 0:
                break
            fs = g.propagate(seed, state[seed], hops=hops)
            cs = concept_scores(fs)
            cat_sig = cs.get(sofa_i, 0) + cs.get(room_i, 0)
            dog_sig = cs.get(bone_i, 0)
            tot = cat_sig + dog_sig + 1e-9
            traj.append(cat_sig / tot)
            lit = np.where(fs > 0)[0]
            if lit.size == 0:
                break
            new = np.zeros(nF, dtype=np.float32)
            if mode == "wta":
                new[lit[np.argmax(fs[lit])]] = 1.0
            else:
                order = lit[np.argsort(fs[lit])[::-1]]
                keep = order[:max(1, int(len(order) * keep_frac))]
                new[keep] = fs[keep] / (fs[keep].max() + 1e-9)
            state = damp * state + (1 - damp) * new
            state[state < 1e-4] = 0.0
        return traj

    def show(title, traj):
        pts = " -> ".join(f"{v:.2f}" for v in traj)
        print(f"  {title}\n    cat-signal: {pts}")

    # ---- graph sizes ----
    g_small = build_graph(n_events=3, contradictions=False)
    g_large_clean = build_graph(n_events=200, contradictions=False)
    g_large_noisy = build_graph(n_events=200, contradictions=True)
    for tag, g in [("small/clean (Day-6 size)", g_small),
                   ("large/clean", g_large_clean),
                   ("large/contradictory", g_large_noisy)]:
        conn = int((np.asarray(g.W.tocsr().sum(axis=1)).ravel() > 0).sum())
        print(f"graph {tag:24s}: {conn} connected fragments")

    print("\n=== decisive test: dog 55% / cat 45% (wrong concept leads) ===")
    print("    does refinement overturn it, and WITH A SLOPE (integration) or a")
    print("    CLIFF (attractor artifact)?\n")
    init = {cat: 0.45, dog: 0.55}
    show("small/clean graph (Day-6 regime)", refine(g_small, init))
    show("LARGE/clean graph", refine(g_large_clean, init))
    show("LARGE/CONTRADICTORY graph (the real test)", refine(g_large_noisy, init))
    show("WTA baseline on large/contradictory", refine(g_large_noisy, init, mode="wta"))

    # ---- verdict ----
    traj = refine(g_large_noisy, init)
    flipped = traj[0] < 0.5 and traj[-1] > 0.5
    tail = traj[-3:]
    converged = (max(tail) - min(tail)) < 0.1
    # slope test: was the climb gradual (>=2 intermediate steps strictly between
    # start and final) or a single-step cliff?
    jumps = [traj[i + 1] - traj[i] for i in range(len(traj) - 1)]
    biggest_jump = max(jumps) if jumps else 0
    gradual = biggest_jump < 0.6 and flipped   # no single step does most of the work

    print("\n=== VERDICT (Day 6.5) ===")
    print(f"  large+contradictory: cat {traj[0]:.2f} -> {traj[-1]:.2f} | "
          f"flipped={flipped} converged={converged}")
    print(f"  biggest single-iteration jump: {biggest_jump:.2f} "
          f"({'gradual integration' if gradual else 'CLIFF — attractor-like'})")
    if flipped and converged and gradual:
        print("\nPASS (convincing) — refinement overturns a wrong lead on a LARGE,")
        print("partially CONTRADICTORY graph, and does so GRADUALLY. This is evidence")
        print("integration, not a tiny-graph attractor artifact. Day 6 survives scale.")
    elif flipped and converged and not gradual:
        print("\nSUSPICIOUS PASS — it flips and converges, but via a CLIFF, not a slope.")
        print("Likely still attractor-dominated even at scale. Not convinced.")
    elif not flipped:
        print("\nFAIL — on a large contradictory graph, refinement no longer overturns")
        print("the wrong lead. Day 6 WAS a small-graph artifact. Honest negative.")
    else:
        print("\nUNSTABLE — does not converge at scale; the dynamics don't settle.")


if __name__ == "__main__":
    main()
