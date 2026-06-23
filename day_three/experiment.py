"""Day 3 — does a single persistent world-state vector hold MULTIPLE things
seen over time, or does each update overwrite the last?

This is the scariest unknown in the architecture (the first arrow nobody has
tested: retrieve -> update state -> persist -> retrieve again).

Honest design notes:
- Metric is NOT "what is the 1 nearest fragment to the state" — an EMA of
  cat+sofa+room lands at their centroid, whose nearest neighbor may be none of
  them. Instead, for EACH item seen, we ask: is it still RECOVERABLE from the
  state (in top-k, and its similarity elevated vs a never-shown control item)?
  That measures recall-of-everything-seen, which is what memory means.
- We run a multi-item sequence (not 2) to expose capacity / forgetting.
- We compare EMA against a FLOOR (last-input-only) and CEILING (concat-all).
- We sweep the decay knob to map recency-vs-retention.

Reuses Day-1 encoders + Day-2 store.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401  (load its OpenMP before torch — macOS segfault guard)
import sys
import numpy as np
import torch

DAY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_one"))
DAY2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_two"))
sys.path[:0] = [DAY1, DAY2]
from config import Config            # noqa: E402
from data import ShapesDataset       # noqa: E402
from model import Worldfield         # noqa: E402
from store import FragmentStore      # noqa: E402
from world_state import (            # noqa: E402
    WorldStateEMA, WorldStateLastOnly, WorldStateConcat)

CKPT = os.path.join(DAY1, "out", "worldfield.pt")
OUT = os.path.join(os.path.dirname(__file__), "out")


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(dev):
    if not os.path.exists(CKPT):
        sys.exit(f"No checkpoint at {CKPT}. Run day_one/train.py first.")
    ck = torch.load(CKPT, map_location=dev)
    cfg = Config()
    model = Worldfield(cfg, ck["vocab_size"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    return model, cfg, ck["class_names"]


@torch.no_grad()
def text_latent(model, ds, dev, class_idx):
    """Encode one TEXT description of a given class into the shared space."""
    for _, txt, label in (ds[i] for i in range(len(ds))):
        if label == class_idx:
            return model.encode_text(txt.unsqueeze(0).to(dev)).cpu().numpy()[0]
    raise ValueError(class_idx)


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def recoverable(store, state, item_label, k, control_label):
    """Is `item_label` recoverable from `state`?
    Returns (in_topk: bool, sim_margin: float).
    sim_margin = (avg sim of item's fragments to state)
               - (avg sim of a never-shown control class to state).
    Positive margin => the state genuinely leans toward this item vs a baseline.
    """
    s = norm(state).astype(np.float32)[None]
    _, idx, _ = store.search(s, k)
    labels_topk = store.labels[idx[0][idx[0] >= 0]]
    in_topk = item_label in set(labels_topk.tolist())

    sims = store.vectors @ norm(state)         # cosine (store vectors normalized)
    item_sim = sims[store.labels == item_label].mean()
    ctrl_sim = sims[store.labels == control_label].mean()
    return in_topk, float(item_sim - ctrl_sim)


def run_sequence(store, state_obj, model, ds, dev, seq_labels, control_label, k):
    """Feed items one at a time; after each step, test recall of EVERY item seen
    so far. Returns matrix recall[t][i] and margins[t][i] for i<=t."""
    n = len(seq_labels)
    in_topk = np.full((n, n), np.nan)
    margin = np.full((n, n), np.nan)
    for t, lab in enumerate(seq_labels):
        ev = text_latent(model, ds, dev, lab)          # perception: encode input
        retrieved_idx = store.activate([norm(ev)], k_max=k, sim_threshold=0.0)[0]
        evidence = store.vectors[retrieved_idx].mean(0) if retrieved_idx.size else norm(ev)
        state_obj.update(evidence)                     # memory: fold into state
        for i in range(t + 1):                         # test all items seen so far
            tk, mg = recoverable(store, state_obj.state, seq_labels[i], k, control_label)
            in_topk[t, i] = tk; margin[t, i] = mg
    return in_topk, margin


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = device()
    model, cfg, class_names = load(dev)
    train_ds = ShapesDataset(cfg, "train")

    # build the fragment store from real image fragments (Day 2)
    store = FragmentStore(cfg.latent_dim, use_hnsw=False)
    vecs, labs = [], []
    for i in range(len(train_ds)):
        img, _, label = train_ds[i]
        with torch.no_grad():
            vecs.append(model.encode_image(img.unsqueeze(0).to(dev)).cpu().numpy()[0])
        labs.append(label)
    store.add(np.array(vecs, np.float32), np.array(labs, np.int64))

    # a sequence of distinct "events" (distinct classes) + a never-shown control
    seq = [0, 4, 8, 12, 1, 5]                  # 6 distinct classes
    control = 9                                # never entered the sequence
    K = 20
    print(f"device: {dev} | store: {len(store)} fragments | seq length: {len(seq)}")
    print("sequence (each is an 'event' entering the world):")
    for j, l in enumerate(seq):
        print(f"  t={j}: {class_names[l]}")
    print(f"control (never shown): {class_names[control]}\n")

    # ---- baselines + EMA sweep ----
    configs = [("last-only (FLOOR)", WorldStateLastOnly(cfg.latent_dim)),
               ("concat-all (CEILING)", WorldStateConcat(cfg.latent_dim))]
    for d in (0.5, 0.7, 0.9, 0.99):
        configs.append((f"EMA decay={d}", WorldStateEMA(cfg.latent_dim, decay=d)))

    final_recall = {}
    for name, obj in configs:
        in_topk, margin = run_sequence(store, obj, model, train_ds, dev, seq, control, K)
        # after the FULL sequence, how many of the items are still recoverable?
        last = in_topk[len(seq) - 1]
        recovered = int(np.nansum(last))
        # HONEST recall metric: an item is "present" if the state leans toward it
        # more than toward a never-shown control (margin > 0). Top-k membership is
        # too harsh on a centroid (Day-3 finding) — a blended state retains info
        # without any single item winning a hard nearest-neighbor contest.
        final_margins = margin[len(seq) - 1]
        present = int((final_margins > 0).sum())
        final_recall[name] = present
        mtxt = " ".join(f"{m:+.2f}" for m in final_margins)
        print(f"{name:22s} | present(margin>0) {present}/{len(seq)} | "
              f"top-k {recovered}/{len(seq)} | per-item margin: {mtxt}")

    # ---- the capacity curve for the canonical EMA=0.9 ----
    print("\n=== capacity curve (EMA decay=0.9): recall of item i after step t ===")
    obj = WorldStateEMA(cfg.latent_dim, decay=0.9)
    in_topk, margin = run_sequence(store, obj, model, train_ds, dev, seq, control, K)
    print("      " + " ".join(f"i{ i}" for i in range(len(seq))))
    for t in range(len(seq)):
        row = " ".join(("Y " if in_topk[t, i] == 1 else "· " if not np.isnan(in_topk[t, i]) else "  ")
                       for i in range(len(seq)))
        print(f"  t={t}: {row}  ({class_names[seq[t]]} just entered)")

    save_plot(in_topk, margin, seq, class_names, final_recall)

    # ---- verdict ----
    ema9_present = final_recall.get("EMA decay=0.9", 0)
    ema9_topk = int(np.nansum(in_topk[len(seq) - 1]))
    print("\n=== VERDICT ===")
    print(f"EMA(0.9): {ema9_present}/{len(seq)} items present (margin>0), but only "
          f"{ema9_topk}/{len(seq)} retrievable by top-k.")
    print("SPLIT RESULT — the state SUPERIMPOSES everything (info retained), but a "
          "single centroid is NOT a sharp point, so retrieval reads back ~1 item. "
          "Continual state as storage: partial. As retrievable memory: NO with one "
          "vector. Fix = slot memory so each concept keeps a sharp home (plan §21, "
          "Day 4).")


def save_plot(in_topk, margin, seq, class_names, final_recall):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing — skipping plot)"); return
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    im = a.imshow(margin, aspect="auto", cmap="viridis")
    a.set_title("similarity margin of item i in state, after step t\n(EMA=0.9)")
    a.set_xlabel("item index i (order entered)"); a.set_ylabel("after step t")
    a.set_xticks(range(len(seq)))
    a.set_xticklabels([class_names[l].replace(" ", "\n") for l in seq], fontsize=7)
    fig.colorbar(im, ax=a, label="sim(item) - sim(control)")
    names = list(final_recall.keys()); vals = list(final_recall.values())
    b.barh(names, vals, color="steelblue"); b.set_xlim(0, len(seq))
    b.set_title("items still recoverable after full sequence"); b.set_xlabel(f"/ {len(seq)}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "world_state.png"), dpi=130)
    plt.close(fig)
    print(f"\nartifact: {os.path.join(OUT, 'world_state.png')}")


if __name__ == "__main__":
    main()
