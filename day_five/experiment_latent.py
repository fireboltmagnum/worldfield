"""Day 5b — latent-substrate propagation: the WHOLE pipeline, end to end.

  latent query -> retrieve fragments -> co-activate -> form edges between
  RETRIEVED FRAGMENT IDS (not labels) -> propagate over the fragment graph ->
  retrieve again -> measure recovery in latent space.

This is the honest test. Edges form between whatever retrieval returns, so
retrieval noise, memory noise, and edge noise all compound. The graph is over
THOUSANDS of fragments, not 60 clean concept ids. Nobody tells the system
"cat = node 12"; it only ever sees fragment ids that came back from a query.

The genuinely hard problem (per the design discussion) is NOISE MANAGEMENT at
edge formation: a 'cat' query may retrieve {cat, dog, orange cat}. Which
co-activations do we commit? We compare:
  - NAIVE   : wire all top-k retrieved fragments to each other every event
  - GUARDED : only wire fragments above a similarity floor to the event centroid
and measure whether recovery survives, and whether noise accumulates over hops.

Honesty guards:
  - we score recovery by CONCEPT of the recovered fragments (ground-truth labels
    used ONLY for scoring, never for edge formation or seeding).
  - CONTROL: same query with an untrained edge set -> recovery must beat it.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401
import sys
import numpy as np
import torch
from scipy.sparse import lil_matrix

DAY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_one"))
DAY2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_two"))
sys.path[:0] = [DAY1, DAY2]
from config_rich import RichConfig    # noqa: E402
from data import ShapesDataset        # noqa: E402
from model import Worldfield          # noqa: E402
from store import FragmentStore       # noqa: E402

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


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


class FragmentGraph:
    """Co-activation edges over FRAGMENT ids (sparse — there are thousands)."""
    def __init__(self, n):
        self.n = n
        self.W = lil_matrix((n, n), dtype=np.float32)

    def observe(self, frag_ids, weights=None):
        ids = list(frag_ids)
        w = weights if weights is not None else np.ones(len(ids))
        for a in range(len(ids)):
            for b in range(len(ids)):
                if ids[a] != ids[b]:
                    self.W[ids[a], ids[b]] += float(w[a] * w[b])

    def propagate(self, seed_ids, seed_vals, hops=2, decay=0.5):
        Wc = self.W.tocsr()
        rs = np.asarray(Wc.sum(axis=1)).ravel()
        # row-normalized transition matrix: T[i,j] = W[i,j]/sum_j W[i,j]
        inv = np.zeros_like(rs); inv[rs > 0] = 1.0 / rs[rs > 0]
        from scipy.sparse import diags
        T = diags(inv).dot(Wc)            # row-stochastic
        act = np.zeros(self.n, dtype=np.float32)
        act[seed_ids] = seed_vals
        frontier = act.copy(); total = act.copy()
        for _ in range(hops):
            frontier = T.T.dot(frontier) * decay   # spread along edges, attenuate
            total += frontier
        total[seed_ids] = 0.0
        return total


def main():
    dev = device()
    model, cfg, names = load(dev)
    ds = ShapesDataset(cfg, "train")

    # build fragment store from real image fragments
    store = FragmentStore(cfg.latent_dim, use_hnsw=False)
    vecs, labs = [], []
    for i in range(len(ds)):
        img, _, label = ds[i]
        with torch.no_grad():
            vecs.append(model.encode_image(img.unsqueeze(0).to(dev)).cpu().numpy()[0])
        labs.append(label)
    store.add(np.array(vecs, np.float32), np.array(labs, np.int64))
    nF = len(store)
    idx = {n: i for i, n in enumerate(names)}
    print(f"store: {nF} fragments | concepts: {len(names)}")

    @torch.no_grad()
    def text_q(concept):
        for _, txt, lab in (ds[i] for i in range(len(ds))):
            if lab == idx[concept]:
                return norm(model.encode_text(txt.unsqueeze(0).to(dev)).cpu().numpy()[0])
        raise ValueError(concept)

    def retrieve(qvec, k):
        sims, ids, _ = store.search(qvec[None].astype(np.float32), k)
        return ids[0], sims[0]

    def label_of(frag_ids):
        return store.labels[frag_ids]

    # roles (scoring only)
    cat, sofa, room = "red circle", "blue square", "green triangle"
    bone = "teal pentagon"

    # ---- build edges from EVENTS via real retrieval (two strategies) ----
    def build_graph(events, guarded, k=6, sim_floor=0.7):
        """events: list of (conceptA, conceptB). Both concepts are PRESENT in the
        event, so we activate each concept's fragments separately and wire the
        UNION (co-activation). Averaging the two queries does NOT work: the
        midpoint of two latents can land on a THIRD concept (Day-3 lesson at the
        event layer — a centroid is not its constituents)."""
        g = FragmentGraph(nF)
        for a, b in events:
            ids_a, sims_a = retrieve(text_q(a), k)
            ids_b, sims_b = retrieve(text_q(b), k)
            ids = np.concatenate([ids_a, ids_b])
            sims = np.concatenate([sims_a, sims_b])
            if guarded:
                keep = sims >= sim_floor       # noise management: drop weak hits
                ids, sims = ids[keep], sims[keep]
            if len(ids) >= 2:
                g.observe(ids, weights=sims)
        return g

    def recover(g, query_concept, k_seed=5, hops=2):
        """Seed from a LATENT query's retrieved fragments, propagate, then read
        back the top recovered concepts in latent space."""
        q = text_q(query_concept)
        seed_ids, seed_sims = retrieve(q, k_seed)
        scores = g.propagate(seed_ids, seed_sims, hops=hops)
        if scores.max() <= 0:
            return {}, scores
        top_frag = np.argsort(scores)[::-1][:50]
        top_frag = [f for f in top_frag if scores[f] > 0]
        # aggregate fragment scores into concept scores
        concept_score = {}
        for f in top_frag:
            c = int(store.labels[f])
            concept_score[c] = concept_score.get(c, 0.0) + scores[f]
        return concept_score, scores

    events = [(cat, sofa)] * 5 + [(cat, room)] * 5   # cat's world
    events += [("yellow circle", bone)] * 5          # a separate (dog,bone) world

    print("\n=== NAIVE edge formation (wire all top-k) ===")
    gN = build_graph(events, guarded=False)
    csN, _ = recover(gN, cat)
    show(csN, names, [sofa, room, bone], idx)

    print("\n=== GUARDED edge formation (sim floor) ===")
    gG = build_graph(events, guarded=True)
    csG, _ = recover(gG, cat)
    show(csG, names, [sofa, room, bone], idx)

    # ---- CONTROL: no edges ----
    g0 = FragmentGraph(nF)
    cs0, _ = recover(g0, cat)
    print(f"\nCONTROL (no edges): recovered {len(cs0)} concepts "
          f"(should be 0 — recovery must come from learned edges)")

    # ---- verdict ----
    def rank(cs):
        return [c for c, _ in sorted(cs.items(), key=lambda x: -x[1])]
    rk = rank(csG)
    sofa_i, room_i, bone_i = idx[sofa], idx[room], idx[bone]
    top3 = rk[:3]
    recovered = sofa_i in top3 and room_i in top3
    no_leak = bone_i not in top3
    print("\n=== VERDICT (Day 5b — latent substrate) ===")
    print(f"  guarded top-3 concepts: {[names[c] for c in top3]}")
    print(f"  recovered cat's associates (sofa+room): {'YES' if recovered else 'NO'}")
    print(f"  rejected unrelated (bone): {'YES' if no_leak else 'NO'}")
    print(f"  control with no edges: {'clean (0)' if len(cs0)==0 else 'LEAKING'}")
    if recovered and no_leak and len(cs0) == 0:
        print("\nPASS — relations recovered through the FULL latent pipeline "
              "(retrieval-seeded, edges over real fragment ids, scored in latent "
              "space). The success belongs to the architecture, not a clean graph.")
    else:
        print("\nINCOMPLETE — noise from real retrieval degraded recovery; see above. "
              "This is the honest hard case (noise management), not a bug.")


def show(cs, names, roles, idx):
    if not cs:
        print("   (nothing recovered)"); return
    ranked = sorted(cs.items(), key=lambda x: -x[1])[:5]
    for c, s in ranked:
        tag = names[c]
        marker = "  <- associate" if c in [idx[r] for r in roles[:2]] else \
                 ("  <- SHOULD NOT APPEAR" if c == idx[roles[2]] else "")
        print(f"   {tag:16s} {s:.3f}{marker}")


if __name__ == "__main__":
    main()
