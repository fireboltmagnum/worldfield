"""Day 4.5 Step A — make slot memory suffer on CONTROLLED geometry.

We don't use the learned encoders here: we synthesize concept 'centers' as
vectors with controllable pairwise similarity, sample fragments around them, and
push the slot memory through the four scenarios that a real memory must survive:

  1. over-capacity     : more concepts than slots — graceful degradation?
  2. confusable        : near-duplicate concepts (cos ~0.9) — separate or collapse?
  3. eviction          : 100 concepts, 8 slots — does LRU actually work?
  4. return-after-gap  : concept appears, leaves, returns — recovered?

CRITICAL metric fix (the Day-4 suspicion): we score retrieval STRICTLY.
  - strict  : the concept is the TOP-1 nearest of some slot (its own sharp home)
  - generous: the concept appears anywhere in some slot's top-k (the inflated
              Day-4 metric — kept only to SHOW the inflation gap)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_two")))
from store import FragmentStore       # noqa: E402
from slot_memory import SlotMemory, _norm  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")
DIM = 128
RNG = np.random.RandomState(0)


def make_concepts(n, n_clusters=None, intra_cos=0.9):
    """Return (centers[n,DIM], group_id[n]). If n_clusters given, concepts come in
    near-duplicate groups with target cosine ~intra_cos within a group."""
    if n_clusters is None:
        c = RNG.randn(n, DIM).astype(np.float32)
        return c / np.linalg.norm(c, axis=1, keepdims=True), np.arange(n)
    centers, groups = [], []
    per = int(np.ceil(n / n_clusters))
    for g in range(n_clusters):
        base = _norm(RNG.randn(DIM).astype(np.float32))
        for _ in range(per):
            if len(centers) >= n:
                break
            # mix base with small noise to hit ~intra_cos within the group
            noise = _norm(RNG.randn(DIM).astype(np.float32))
            v = intra_cos * base + np.sqrt(max(1 - intra_cos**2, 0)) * noise
            centers.append(_norm(v)); groups.append(g)
    return np.array(centers, np.float32), np.array(groups)


def build_store(centers, frags_per=80, jitter=0.05):
    store = FragmentStore(DIM, use_hnsw=False)
    vecs, labs = [], []
    for ci, c in enumerate(centers):
        f = c[None] + jitter * RNG.randn(frags_per, DIM).astype(np.float32)
        vecs.append(f); labs += [ci] * frags_per
    store.add(np.vstack(vecs), np.array(labs, np.int64))
    return store


def evidence_for(store, concept_label, k=20):
    """Simulate perception: retrieve the concept's fragments, average them."""
    idx = np.where(store.labels == concept_label)[0]
    pick = RNG.choice(idx, size=min(k, len(idx)), replace=False)
    return store.vectors[pick].mean(0)


def score(store, mem, concept_labels, k):
    """For each concept, is it recoverable from the slots?
    Returns (strict_count, generous_count)."""
    slots = mem.active_slots()
    strict = generous = 0
    if slots.shape[0] == 0:
        return 0, 0
    # precompute each slot's top-1 and top-k class sets
    slot_top1, slot_topk = [], []
    for s in slots:
        _, idx, _ = store.search(_norm(s).astype(np.float32)[None], k)
        row = store.labels[idx[0][idx[0] >= 0]]
        slot_top1.append(int(row[0]) if len(row) else -1)
        slot_topk.append(set(row.tolist()))
    top1_set = set(slot_top1)
    for lab in concept_labels:
        if lab in top1_set:
            strict += 1
        if any(lab in tk for tk in slot_topk):
            generous += 1
    return strict, generous


def run_sequence(store, mem, seq, k=20):
    for lab in seq:
        mem.update(evidence_for(store, lab, k))


def scenario(title, n_concepts, n_slots, n_clusters=None, intra_cos=0.9,
             seq=None, k=20, merge_threshold=0.6):
    centers, groups = make_concepts(n_concepts, n_clusters, intra_cos)
    store = build_store(centers)
    mem = SlotMemory(DIM, n_slots=n_slots, decay=0.5, merge_threshold=merge_threshold)
    if seq is None:
        seq = list(range(n_concepts))
    run_sequence(store, mem, seq, k)
    # we evaluate recall over the LAST n_slots distinct concepts that were shown
    # (anything older than capacity is expected to be evictable)
    distinct_in_order = list(dict.fromkeys(seq))
    strict, generous = score(store, mem, distinct_in_order, k)
    n_eval = len(distinct_in_order)
    print(f"  {title}")
    print(f"    concepts={n_concepts} slots={n_slots} used={int(mem.used.sum())} "
          f"| evaluated {n_eval} shown concepts")
    print(f"    strict (own top-1 slot): {strict}/{n_eval}   "
          f"generous (in some top-k): {generous}/{n_eval}   "
          f"INFLATION gap: {generous - strict}")
    return strict, generous, n_eval


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Day 4.5 Step A: slot memory under controlled-geometry stress ===")
    print("(strict = concept is some slot's TOP-1; generous = anywhere in top-k)\n")

    print("1. OVER-CAPACITY (more concepts than slots)")
    scenario("8 slots, 8 concepts (baseline)", 8, 8)
    scenario("8 slots, 20 concepts", 20, 8)
    scenario("8 slots, 40 concepts", 40, 8)

    print("\n2. CONFUSABLE near-duplicate concepts (intra-group cos ~0.9)")
    scenario("8 slots, 8 concepts in 2 tight groups", 8, 8, n_clusters=2, intra_cos=0.9)
    scenario("8 slots, 8 concepts, 1 group cos~0.95", 8, 8, n_clusters=1, intra_cos=0.95)
    # routing threshold matters: if merge_threshold < intra_cos, duplicates merge
    scenario("same, merge_threshold=0.85 (split harder)", 8, 8, n_clusters=2,
             intra_cos=0.9, merge_threshold=0.85)

    print("\n3. EVICTION pressure (100 concepts streamed through 8 slots)")
    s, g, n = scenario("8 slots, 100 concepts (eval all 100)", 100, 8)
    # the honest question: are the MOST RECENT 8 concepts retained?
    centers, _ = make_concepts(100)
    store = build_store(centers)
    mem = SlotMemory(DIM, n_slots=8, decay=0.5, merge_threshold=0.6)
    run_sequence(store, mem, list(range(100)))
    recent = list(range(92, 100))
    sr, gr = score(store, mem, recent, 20)
    print(f"    of the LAST 8 concepts seen: strict {sr}/8  generous {gr}/8 "
          f"(graceful = recent retained, old evicted)")

    print("\n4. RETURN-AFTER-ABSENCE (concept appears, leaves, returns)")
    centers, _ = make_concepts(20)
    store = build_store(centers)
    mem = SlotMemory(DIM, n_slots=8, decay=0.5, merge_threshold=0.6)
    # show concept 0, then flood with 0..15 (evicts it), then show 0 again
    mem.update(evidence_for(store, 0))
    run_sequence(store, mem, list(range(1, 16)))   # 15 others -> evicts concept 0
    sr_before, _ = score(store, mem, [0], 20)
    mem.update(evidence_for(store, 0))             # it returns
    sr_after, _ = score(store, mem, [0], 20)
    print(f"  concept 0 strict-recoverable: after eviction={sr_before}/1  "
          f"after it returns={sr_after}/1  (should go 0 -> 1)")

    print("\n=== READING ===")
    print("Look at the INFLATION gap: if generous >> strict, Day-4's 6/6 was a")
    print("metric artifact. If strict tracks slot count and recent-concept recall")
    print("holds while old concepts evict, the slot architecture is honestly sound.")


if __name__ == "__main__":
    main()
