"""Day 5c — reasoning under AMBIGUOUS retrieval (the genuinely hard case).

Day 5b worked because seeds were perfect: 'cat' query -> 5 cat fragments. Real
systems get contaminated seeds: 'cat' -> {cat, dog, orange-cat, ...}. The open
question is what propagation does with a noisy seed:

    Does noise AMPLIFY across hops, or CANCEL?
    Can the system still converge to the correct associate?
    Do MORE hops help or hurt?

We test two ways:
  (A) CONTROLLED contamination: build the seed as a mix of correct + wrong
      concept fragments at a dial-able ratio, sweep it, and plot the response
      curve. This is the instrument — it measures amplify-vs-cancel precisely.
  (B) REALISTIC ambiguity: query with a genuinely confusable concept (one of the
      near-duplicate blues) and let retrieval contaminate the seed on its own.

Edges are built once from clean events (the world the system has learned). Only
the QUERY/seed is noisy at test time — exactly the real situation.
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
sys.path[:0] = [DAY1, DAY2]
from config_rich import RichConfig    # noqa: E402
from data import ShapesDataset        # noqa: E402
from model import Worldfield          # noqa: E402
from store import FragmentStore       # noqa: E402
from experiment_latent import FragmentGraph  # noqa: E402  (reuse the graph)

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

    def retrieve(qvec, k):
        s, i, _ = store.search(qvec[None].astype(np.float32), k)
        return i[0], s[0]

    def frags_of(concept, k):
        return retrieve(text_q(concept), k)

    # roles
    cat, sofa, room = "red circle", "blue square", "green triangle"
    dog, bone = "yellow circle", "teal pentagon"

    # ---- build the world's edges from CLEAN events (learned once) ----
    g = FragmentGraph(nF)
    for _ in range(5):
        for (a, b) in [(cat, sofa), (cat, room), (dog, bone)]:
            ia, sa = frags_of(a, 6); ib, sb = frags_of(b, 6)
            ids = np.concatenate([ia, ib]); w = np.concatenate([sa, sb])
            g.observe(ids, weights=w)

    def recover_rank(seed_ids, seed_vals, hops):
        scores = g.propagate(seed_ids, seed_vals, hops=hops)
        if scores.max() <= 0:
            return []
        per_concept = {}
        for f in np.argsort(scores)[::-1][:60]:
            if scores[f] <= 0:
                break
            c = int(store.labels[f]); per_concept[c] = per_concept.get(c, 0) + scores[f]
        return [c for c, _ in sorted(per_concept.items(), key=lambda x: -x[1])]

    sofa_i, room_i, bone_i = idx[sofa], idx[room], idx[bone]

    def correct(rank):
        """cat's true associates are sofa & room; success = both in top-2,
        and bone (dog's world) NOT in top-2."""
        top2 = rank[:2]
        return (sofa_i in top2 and room_i in top2) and (bone_i not in top2)

    # ================= (A) CONTROLLED contamination sweep =================
    print("=== (A) controlled contamination: seed = cat + (frac) dog ===")
    print("    measuring whether wrong-concept noise in the SEED breaks recovery")
    print(f"    {'contam':>7} | {'hops=1':>18} | {'hops=2':>18} | {'hops=3':>18}")
    k = 6
    cat_ids, cat_s = frags_of(cat, k)
    dog_ids, dog_s = frags_of(dog, k)
    rows = []
    for frac in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        # seed: keep cat fragments at weight (1-frac), dog at weight frac
        seed_ids = np.concatenate([cat_ids, dog_ids])
        seed_vals = np.concatenate([cat_s * (1 - frac), dog_s * frac]).astype(np.float32)
        cells = []
        for hops in (1, 2, 3):
            rank = recover_rank(seed_ids, seed_vals, hops)
            top2 = [names[c] for c in rank[:2]] if rank else []
            ok = "OK " if correct(rank) else "xx "
            cells.append(f"{ok}{','.join(t.split()[0] for t in top2):>14}")
        rows.append((frac, cells))
        print(f"    {frac:>7.1f} | {cells[0]:>18} | {cells[1]:>18} | {cells[2]:>18}")

    # find the breaking point at hops=2
    break_frac = None
    for frac, _ in rows:
        seed_ids = np.concatenate([cat_ids, dog_ids])
        seed_vals = np.concatenate([cat_s * (1 - frac), dog_s * frac]).astype(np.float32)
        if not correct(recover_rank(seed_ids, seed_vals, 2)):
            break_frac = frac; break
    print(f"    -> recovery breaks at contamination >= "
          f"{break_frac if break_frac is not None else '>1.0 (never)'} (hops=2)")

    # ================= amplify vs cancel: does hops help or hurt? =========
    print("\n=== does propagation AMPLIFY or CANCEL seed noise? ===")
    frac = 0.4   # contaminated but cat still majority
    seed_ids = np.concatenate([cat_ids, dog_ids])
    seed_vals = np.concatenate([cat_s * (1 - frac), dog_s * frac]).astype(np.float32)
    for hops in (1, 2, 3, 4):
        rank = recover_rank(seed_ids, seed_vals, hops)
        bone_pos = rank.index(bone_i) + 1 if bone_i in rank else None
        print(f"    hops={hops}: top3={[names[c].split()[0] for c in rank[:3]]} | "
              f"bone(noise) rank={bone_pos} | recovery {'OK' if correct(rank) else 'broken'}")

    # ================= (B) REALISTIC confusable query =====================
    print("\n=== (B) realistic ambiguity: query a confusable blue ===")
    # build a world where 'navy square' has an associate, then query with the
    # genuinely confusable 'navy square' and see if retrieval contamination
    # (royalblue/blue/skyblue) derails recovery.
    g2 = FragmentGraph(nF)
    navy, target = "navy square", "olive circle"
    for _ in range(5):
        ia, sa = frags_of(navy, 6); ib, sb = frags_of(target, 6)
        g2.observe(np.concatenate([ia, ib]), weights=np.concatenate([sa, sb]))
    seed_ids2, seed_s2 = frags_of(navy, 6)
    seed_labels = [names[store.labels[i]] for i in seed_ids2]
    print(f"    'navy square' seed actually retrieved: "
          f"{[s.split()[0]+' '+s.split()[1] for s in seed_labels]}")
    scores = g2.propagate(seed_ids2, seed_s2, hops=2)
    per = {}
    for f in np.argsort(scores)[::-1][:60]:
        if scores[f] <= 0:
            break
        c = int(store.labels[f]); per[c] = per.get(c, 0) + scores[f]
    rank2 = [c for c, _ in sorted(per.items(), key=lambda x: -x[1])]
    print(f"    recovered top-3: {[names[c] for c in rank2[:3]]}")
    realistic_ok = idx[target] in rank2[:2]
    print(f"    recovered correct associate '{target}': {'YES' if realistic_ok else 'NO'}")

    print("\n=== VERDICT (Day 5c) ===")
    robust = (break_frac is None) or (break_frac >= 0.5)
    print(f"  tolerates seed contamination up to "
          f"{'>=50%' if robust else f'<{break_frac:.0%}'} before breaking")
    print(f"  realistic confusable-query recovery: {'YES' if realistic_ok else 'NO'}")
    if robust and realistic_ok:
        print("\nPASS — reasoning survives contaminated seeds: as long as the correct")
        print("concept stays plurality, propagation CANCELS minority noise rather than")
        print("amplifying it. Graceful degradation, not collapse.")
    elif robust or realistic_ok:
        print("\nPARTIAL — robust in one regime, fragile in the other; see above.")
    else:
        print("\nFAIL — seed noise propagates and derails recovery. Noise management")
        print("at the seed stage is required before this counts as reasoning.")


if __name__ == "__main__":
    main()
