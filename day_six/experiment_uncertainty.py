"""Day 6.6 — can Worldfield REPRESENT uncertainty, or does it always collapse?

Day 6.5 was a warning: refinement pinned to EXACTLY 1.00, washing out a 10%
minority. That's resolution, not reasoning. A scientist faced with balanced
evidence says 'A=50%, B=50%, both still live' — an attractor system says
'A=100%'. This is the line between an attractor and a probabilistic reasoner.

Tests:
  1. BALANCED evidence (cat--room 50% / cat--bone 50%): does it hold ~50/50 or
     collapse to one winner?
  2. GRADED evidence (70/30, 60/40): does the final state TRACK the input ratio
     (probabilistic) or snap to 100/0 (attractor)?
  3. SYMMETRY-BREAK probe: start exactly balanced, nudge 1%. A probabilistic
     system stays ~51/49; an attractor SNAPS to 100/0. This distinguishes
     'preserving uncertainty' from merely 'stuck at the symmetric fixed point'.

We compare the Day-6 hard-pruning loop (keep_frac + max-normalize, which forces a
winner) against a SOFTENED loop (no winner-take pruning, sum-normalized like a
distribution) to see whether uncertainty collapse is fundamental or just a
consequence of the update rule.
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
    by_concept = {c: np.where(store.labels == c)[0] for c in range(len(names))}

    def sample(cn, k):
        return RNG.choice(by_concept[idx[cn]], size=min(k, len(by_concept[idx[cn]])),
                          replace=False)

    cat, room, bone = "red circle", "green triangle", "teal pentagon"
    room_i, bone_i = idx[room], idx[bone]

    def build_graph(p_room, p_bone, n_events=200, fp=8):
        """cat co-occurs with room with prob p_room, with bone with prob p_bone.
        This bakes a GRADED relational structure: cat's evidence is split."""
        g = FragmentGraph(nF)
        for _ in range(n_events):
            ic = sample(cat, fp)
            if RNG.rand() < p_room:
                ir = sample(room, fp)
                g.observe(np.concatenate([ic, ir]), np.ones(len(ic) + len(ir), np.float32))
            if RNG.rand() < p_bone:
                ib = sample(bone, fp)
                g.observe(np.concatenate([ic, ib]), np.ones(len(ic) + len(ib), np.float32))
        return g

    def cscore(fs):
        per = {}
        for f in np.argsort(fs)[::-1][:400]:
            if fs[f] <= 0:
                break
            c = int(store.labels[f]); per[c] = per.get(c, 0.0) + fs[f]
        return per

    # the graph's backbone: only fragments WITH edges can carry activation across
    # iterations. The soft update must stay on this backbone, or it diffuses into
    # dead (unconnected) fragments and the signal dies (a real constraint the
    # substrate imposes — discovered debugging the first soft-rule version).
    def backbone_of(g):
        return np.asarray(g.W.tocsr().sum(axis=1)).ravel() > 0

    def diffuse(g, state, hops=2, decay=0.5):
        """A DIFFUSION step (distinct from Day-5b's retrieval-style propagate).
        Key difference: it does NOT zero the seed — it spreads mass while KEEPING
        existing mass, so a dense distribution-state survives iteration. Day-5b's
        'don't return the seed' rule is correct for one-shot recovery but fatal
        for iterative uncertainty-passing (it annihilates a dense state)."""
        Wc = g.W.tocsr()
        rs = np.asarray(Wc.sum(axis=1)).ravel()
        inv = np.zeros_like(rs); inv[rs > 0] = 1.0 / rs[rs > 0]
        from scipy.sparse import diags
        T = diags(inv).dot(Wc)            # row-stochastic transition
        x = state.copy()
        out = state.copy()
        for _ in range(hops):
            x = T.T.dot(x) * decay
            out = out + x                  # accumulate, keep prior mass
        return out

    def refine(g, iters=8, hops=2, damp=0.6, rule="hard"):
        """rule='hard' : Day-6 winner-forcing update (keep top 50%, max-normalize)
           rule='soft' : distribution-like update over the connected backbone
                         (sum-normalized, no winner pruning, no hard floor)."""
        conn = backbone_of(g)
        state = np.zeros(nF, dtype=np.float32)
        state[sample(cat, 8)] = 1.0
        traj = []
        for _ in range(iters):
            seed = np.where(state > 0)[0]
            if seed.size == 0:
                break
            if rule == "hard":
                fs = g.propagate(seed, state[seed], hops=hops)   # retrieval-style
            else:
                fs = diffuse(g, state, hops=hops)                # diffusion-style
            cs = cscore(fs)
            r, b = cs.get(room_i, 0.0), cs.get(bone_i, 0.0)
            tot = r + b + 1e-9
            traj.append(r / tot)
            lit = np.where(fs > 0)[0]
            if lit.size == 0:
                break
            new = np.zeros(nF, dtype=np.float32)
            if rule == "hard":
                order = lit[np.argsort(fs[lit])[::-1]]
                keep = order[:max(1, int(len(order) * 0.5))]
                new[keep] = fs[keep] / (fs[keep].max() + 1e-9)   # max-norm: forces a peak
            else:
                lit_c = lit[conn[lit]]
                if lit_c.size == 0:
                    break
                new[lit_c] = fs[lit_c] / (fs[lit_c].sum() + 1e-9)  # distribution
            state = damp * state + (1 - damp) * new
            if rule == "soft":
                state[~conn] = 0.0
        return traj

    def show(t, label):
        print(f"    {label:28s}: " + " -> ".join(f"{v:.2f}" for v in t))

    print("=== 1. BALANCED evidence: cat--room 50% / cat--bone 50% ===")
    print("    (room-signal fraction; 0.50 = uncertainty preserved, 1.00/0.00 = collapse)")
    g_bal = build_graph(0.5, 0.5)
    show(refine(g_bal, rule="hard"), "hard update (Day-6 rule)")
    show(refine(g_bal, rule="soft"), "soft update (distribution)")

    print("\n=== 2. GRADED evidence: does final state TRACK the input ratio? ===")
    print("    input room:bone  ->  final room-signal (probabilistic = tracks input)")
    for p_room, p_bone in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)]:
        g = build_graph(p_room, p_bone)
        th = refine(g, rule="hard")[-1]
        ts = refine(g, rule="soft")[-1]
        print(f"    {p_room:.1f}:{p_bone:.1f}  ->  hard={th:.2f}   soft={ts:.2f}")

    print("\n=== 3. SYMMETRY-BREAK: start balanced, results should reflect rule ===")
    g_sym = build_graph(0.5, 0.5)
    hard_final = refine(g_sym, rule="hard")[-1]
    soft_final = refine(g_sym, rule="soft")[-1]

    # ---- verdict ----
    # probabilistic = soft update tracks the input ratio AND holds ~0.5 on balanced;
    # attractor    = hard update collapses balanced evidence to 0/1.
    print("\n=== VERDICT (Day 6.6) ===")
    print(f"  balanced evidence: hard rule -> {hard_final:.2f}, soft rule -> {soft_final:.2f}")
    # measure tracking for the soft rule across the graded sweep
    softs = [refine(build_graph(pr, pb), rule="soft")[-1]
             for pr, pb in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)]]
    monotone = all(softs[i] >= softs[i + 1] - 0.05 for i in range(len(softs) - 1))
    spread = max(softs) - min(softs)
    hard_collapses = abs(hard_final - 0.5) > 0.35
    soft_preserves = abs(soft_final - 0.5) < 0.20
    print(f"  soft rule across 70/30..30/70: {[round(s,2) for s in softs]} "
          f"(monotone={monotone}, spread={spread:.2f})")
    if hard_collapses and soft_preserves and monotone and spread > 0.15:
        print("\nKEY FINDING — uncertainty collapse is NOT fundamental: it is caused by")
        print("the winner-forcing UPDATE RULE. With a distribution-style update, the")
        print("substrate PRESERVES graded uncertainty and TRACKS input evidence ratios.")
        print("Worldfield can be an attractor OR a probabilistic reasoner — the update")
        print("rule decides which. This is the answer to 'can it represent uncertainty?'")
    elif soft_preserves and not monotone:
        print("\nPARTIAL — soft rule avoids total collapse but does not cleanly track")
        print("input ratios; uncertainty is preserved but not calibrated.")
    elif not soft_preserves:
        print("\nUNCERTAINTY COLLAPSES regardless of update rule — the collapse is")
        print("fundamental to this substrate/propagation, not just the update. Honest")
        print("negative: Worldfield is an attractor system, not a probabilistic one.")
    else:
        print("\nMIXED — see numbers above.")


if __name__ == "__main__":
    main()
