"""Day 2 — does retrieval over the LEARNED latent space return semantically
correct, sparse neighbors, and survive distractors + scale?

This can fail. That's the point. A green checkmark from FAISS alone proves
nothing; the honest signal is whether retrieved fragments share the query's
class, whether a threshold activation rule stays sparse without collapsing, and
whether both hold as we bury the real fragments under random distractors and
scale the store up.

Reuses the Day-1 encoders (out/worldfield.pt) so fragments are real and carry
ground-truth class labels.
"""
import os
# torch and faiss both bundle an OpenMP runtime; on macOS they clash (segfault).
# Importing faiss FIRST makes its runtime load before torch's, which avoids it.
# The env var is a belt-and-suspenders fallback. Both must precede torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401  (imported for side effect: load its OpenMP first)
import sys
import numpy as np
import torch
import torch.nn.functional as F

# reuse Day-1 code
DAY1 = os.path.join(os.path.dirname(__file__), "..", "day_one")
sys.path.insert(0, os.path.abspath(DAY1))
from config import Config            # noqa: E402
from data import ShapesDataset       # noqa: E402
from model import Worldfield         # noqa: E402

from store import FragmentStore      # noqa: E402

CKPT = os.path.join(DAY1, "out", "worldfield.pt")
OUT = os.path.join(os.path.dirname(__file__), "out")


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(dev):
    if not os.path.exists(CKPT):
        sys.exit(f"No checkpoint at {CKPT}. Run day_one/train.py first.")
    ck = torch.load(CKPT, map_location=dev)
    cfg = Config()
    model = Worldfield(cfg, ck["vocab_size"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["class_names"]


@torch.no_grad()
def embed_images(model, ds, dev, n):
    """Produce n real fragments (image latents) with ground-truth class."""
    vecs, labs = [], []
    for i in range(min(n, len(ds))):
        img, _, label = ds[i]
        z = model.encode_image(img.unsqueeze(0).to(dev))
        vecs.append(z.cpu().numpy()[0]); labs.append(label)
    return np.array(vecs, dtype=np.float32), np.array(labs, dtype=np.int64)


@torch.no_grad()
def embed_text_queries(model, ds, dev, n):
    """Held-out TEXT queries (cross-modal: text query -> image fragments)."""
    vecs, labs = [], []
    for i in range(min(n, len(ds))):
        _, txt, label = ds[i]
        z = model.encode_text(txt.unsqueeze(0).to(dev))
        vecs.append(z.cpu().numpy()[0]); labs.append(label)
    return np.array(vecs, dtype=np.float32), np.array(labs, dtype=np.int64)


def precision_at_k(store, q_vecs, q_labels, k):
    """Of the top-k retrieved fragments, what fraction share the query class?"""
    _, idx, lat = store.search(q_vecs, k)
    hits, total = 0, 0
    for row, ql in zip(idx, q_labels):
        row = row[row >= 0]
        hits += int((store.labels[row] == ql).sum())
        total += len(row)
    return hits / max(total, 1), lat


def run():
    os.makedirs(OUT, exist_ok=True)
    dev = device()
    model, cfg, class_names = load_model(dev)
    print(f"device: {dev} | latent_dim: {cfg.latent_dim} | classes: {len(class_names)}")

    # real fragments from the train split; held-out text queries from val split
    train_ds = ShapesDataset(cfg, "train")
    val_ds = ShapesDataset(cfg, "val")
    real_vecs, real_labs = embed_images(model, train_ds, dev, n=len(train_ds))
    q_vecs, q_labs = embed_text_queries(model, val_ds, dev, n=600)
    print(f"real fragments: {len(real_vecs)} | text queries (held-out): {len(q_vecs)}\n")

    chance = 1.0 / len(class_names)
    rng = np.random.RandomState(0)
    K = 10

    # ---- Experiment 1: semantic precision@k, exact vs HNSW ----
    print("=== 1. semantic retrieval (text query -> image fragments) ===")
    for use_hnsw in (False, True):
        store = FragmentStore(cfg.latent_dim, use_hnsw=use_hnsw)
        store.add(real_vecs, real_labs)
        p, lat = precision_at_k(store, q_vecs, q_labs, K)
        kind = "HNSW " if use_hnsw else "exact"
        print(f"  {kind} | store={len(store):>6} | precision@{K}={p:.3f} "
              f"(chance {chance:.3f}) | {lat:.3f} ms/query")

    # ---- Experiment 2: distractor robustness ----
    print("\n=== 2. distractors: bury real fragments in random noise ===")
    for n_distract in (0, 10_000, 90_000):
        store = FragmentStore(cfg.latent_dim, use_hnsw=False)
        store.add(real_vecs, real_labs)
        if n_distract:
            d = rng.randn(n_distract, cfg.latent_dim).astype(np.float32)
            store.add(d, np.full(n_distract, -1, dtype=np.int64))  # label -1 = noise
        p, lat = precision_at_k(store, q_vecs, q_labs, K)
        frac = len(real_vecs) / len(store)
        print(f"  store={len(store):>6} ({frac*100:4.1f}% real) | "
              f"precision@{K}={p:.3f} | {lat:.3f} ms/query")

    # ---- Experiment 3: scale latency ----
    print("\n=== 3. scale: latency vs store size (exact vs HNSW) ===")
    for n_total in (1_000, 10_000, 100_000, 500_000):
        for use_hnsw in (False, True):
            store = FragmentStore(cfg.latent_dim, use_hnsw=use_hnsw)
            store.add(real_vecs, real_labs)
            pad = n_total - len(real_vecs)
            if pad > 0:
                store.add(rng.randn(pad, cfg.latent_dim).astype(np.float32),
                          np.full(pad, -1, dtype=np.int64))
            _, lat = precision_at_k(store, q_vecs[:200], q_labs[:200], K)
            print(f"  {'HNSW ' if use_hnsw else 'exact'} | "
                  f"store={len(store):>7} | {lat:.3f} ms/query")

    # ---- Experiment 4: the sparsity question ----
    print("\n=== 4. sparsity: does a threshold activation rule stay small? ===")
    store = FragmentStore(cfg.latent_dim, use_hnsw=False)
    store.add(real_vecs, real_labs)
    print("  (active = retrieved within top-64 AND sim >= threshold)")
    for thr in (0.0, 0.5, 0.8, 0.9, 0.95):
        active = store.activate(q_vecs, k_max=64, sim_threshold=thr)
        sizes = np.array([a.size for a in active])
        # purity: of activated fragments, fraction matching the query class
        pure = []
        for a, ql in zip(active, q_labs):
            if a.size:
                pure.append((store.labels[a] == ql).mean())
        purity = float(np.mean(pure)) if pure else 0.0
        empties = int((sizes == 0).mean() * 100)
        print(f"  thr={thr:.2f} | active/query: mean={sizes.mean():5.1f} "
              f"max={sizes.max():3d} | empty queries={empties:2d}% | "
              f"purity={purity:.3f}")

    # ---- verdict ----
    store = FragmentStore(cfg.latent_dim, use_hnsw=False)
    store.add(real_vecs, real_labs)
    d = rng.randn(90_000, cfg.latent_dim).astype(np.float32)
    store.add(d, np.full(90_000, -1, dtype=np.int64))
    p_stress, _ = precision_at_k(store, q_vecs, q_labs, K)
    print("\n=== VERDICT ===")
    print(f"precision@{K} with 95% distractors: {p_stress:.3f} (chance {chance:.3f})")
    ok = p_stress > 0.7
    print("PASS — semantic retrieval survives distractors; sparsity is tunable. "
          "Build the update step next." if ok else
          "WEAK — retrieval degrades under distractors; fix before building update.")


if __name__ == "__main__":
    run()
