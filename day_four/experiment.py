"""Day 4 — does slot memory recover what one vector could not?

Same capacity test as Day 3, same store, same sequence. The only change: replace
the single EMA vector with K slots + routing. Hypothesis: items that collapsed
into one centroid (Day 3: ~1 retrievable) now keep sharp homes and become
individually retrievable.

Slot-aware metric: an item is RETRIEVABLE if ANY slot returns it in top-k, and
PRESENT if ANY slot's margin over a never-shown control is > 0. We compare slot
memory directly against the Day-3 single-vector EMA on identical inputs.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401  (load OpenMP before torch — macOS segfault guard)
import sys
import numpy as np
import torch

DAY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_one"))
DAY2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_two"))
DAY3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_three"))
sys.path[:0] = [DAY1, DAY2, DAY3]
from config import Config            # noqa: E402
from data import ShapesDataset       # noqa: E402
from model import Worldfield         # noqa: E402
from store import FragmentStore      # noqa: E402
from world_state import WorldStateEMA  # noqa: E402  (Day-3 baseline)
from slot_memory import SlotMemory, _norm  # noqa: E402

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
    for _, txt, label in (ds[i] for i in range(len(ds))):
        if label == class_idx:
            return model.encode_text(txt.unsqueeze(0).to(dev)).cpu().numpy()[0]
    raise ValueError(class_idx)


def item_present_singlevec(store, state, item_label, control_label):
    sims = store.vectors @ _norm(state)
    return float(sims[store.labels == item_label].mean()
                 - sims[store.labels == control_label].mean())


def item_retrievable_singlevec(store, state, item_label, k):
    _, idx, _ = store.search(_norm(state).astype(np.float32)[None], k)
    return item_label in set(store.labels[idx[0][idx[0] >= 0]].tolist())


def item_present_slots(store, slots, item_label, control_label):
    """Best margin across all slots — an item is present if ANY slot leans to it."""
    if slots.shape[0] == 0:
        return -1.0
    best = -9.9
    for s in slots:
        sims = store.vectors @ _norm(s)
        m = float(sims[store.labels == item_label].mean()
                  - sims[store.labels == control_label].mean())
        best = max(best, m)
    return best


def item_retrievable_slots(store, slots, item_label, k):
    """Retrievable if ANY slot returns the item in its top-k."""
    for s in slots:
        _, idx, _ = store.search(_norm(s).astype(np.float32)[None], k)
        if item_label in set(store.labels[idx[0][idx[0] >= 0]].tolist()):
            return True
    return False


def run(store, mem, model, ds, dev, seq, control, k, is_slot):
    """Feed sequence; after the full sequence, score every item."""
    for lab in seq:
        ev = text_latent(model, ds, dev, lab)
        retrieved = store.activate([_norm(ev)], k_max=k, sim_threshold=0.0)[0]
        evidence = store.vectors[retrieved].mean(0) if retrieved.size else _norm(ev)
        mem.update(evidence)
    present, retr = [], []
    for lab in seq:
        if is_slot:
            slots = mem.active_slots()
            present.append(item_present_slots(store, slots, lab, control))
            retr.append(item_retrievable_slots(store, slots, lab, k))
        else:
            present.append(item_present_singlevec(store, mem.state, lab, control))
            retr.append(item_retrievable_singlevec(store, mem.state, lab, k))
    return np.array(present), np.array(retr, dtype=bool)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = device()
    model, cfg, class_names = load(dev)
    train_ds = ShapesDataset(cfg, "train")

    store = FragmentStore(cfg.latent_dim, use_hnsw=False)
    vecs, labs = [], []
    for i in range(len(train_ds)):
        img, _, label = train_ds[i]
        with torch.no_grad():
            vecs.append(model.encode_image(img.unsqueeze(0).to(dev)).cpu().numpy()[0])
        labs.append(label)
    store.add(np.array(vecs, np.float32), np.array(labs, np.int64))

    seq = [0, 4, 8, 12, 1, 5]          # 6 distinct concepts (same as Day 3)
    control = 9
    K = 20
    print(f"device: {dev} | store: {len(store)} | seq: {[class_names[l] for l in seq]}")
    print(f"control (never shown): {class_names[control]}\n")

    # ---- baseline: Day-3 single vector ----
    ema = WorldStateEMA(cfg.latent_dim, decay=0.9)
    p_ema, r_ema = run(store, ema, model, train_ds, dev, seq, control, K, is_slot=False)

    # ---- slot memory, swept over slot count ----
    print(f"{'memory':24s} | retrievable (top-k) | present (margin>0)")
    print(f"{'single vector (Day 3)':24s} | "
          f"{int(r_ema.sum())}/{len(seq)}              | {int((p_ema>0).sum())}/{len(seq)}")
    results = {"single vector (Day 3)": int(r_ema.sum())}
    best_slots = None
    for n_slots in (4, 8, 16):
        mem = SlotMemory(cfg.latent_dim, n_slots=n_slots, decay=0.5, merge_threshold=0.6)
        p, r = run(store, mem, model, train_ds, dev, seq, control, K, is_slot=True)
        print(f"{'slot memory ('+str(n_slots)+' slots)':24s} | "
              f"{int(r.sum())}/{len(seq)}              | {int((p>0).sum())}/{len(seq)}")
        results[f"{n_slots} slots"] = int(r.sum())
        if n_slots == 8:
            best_slots = (mem, p, r)

    # per-item detail for 8-slot vs single vector
    mem, p8, r8 = best_slots
    print(f"\nper-item retrievable (8 slots):  "
          + " ".join(f"{class_names[seq[i]].split()[0]}:{'Y' if r8[i] else '.'}"
                     for i in range(len(seq))))
    print(f"per-item retrievable (1 vector): "
          + " ".join(f"{class_names[seq[i]].split()[0]}:{'Y' if r_ema[i] else '.'}"
                     for i in range(len(seq))))
    print(f"slots used: {int(mem.used.sum())}/{mem.n_slots} "
          f"for {len(set(seq))} distinct concepts")

    save_plot(results, len(seq))

    # ---- verdict ----
    slot8 = results.get("8 slots", 0)
    print("\n=== VERDICT ===")
    print(f"single vector: {int(r_ema.sum())}/{len(seq)} retrievable | "
          f"8 slots: {slot8}/{len(seq)} retrievable")
    if slot8 >= len(seq) - 1 and slot8 > int(r_ema.sum()):
        print("PASS — slot memory recovers what one vector could not. NOT compressing "
              "concepts to a single point lets each keep a sharp, retrievable home. "
              "Perception -> retrievable memory works in the same substrate (plan §21).")
    elif slot8 > int(r_ema.sum()):
        print("PARTIAL — slots beat one vector but don't recover all items. Routing or "
              "slot count needs work; the direction is right.")
    else:
        print("WEAK — slots did not beat one vector. Re-examine routing / merge "
              "threshold before trusting layered memory.")


def save_plot(results, n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing — skipping plot)"); return
    names = list(results.keys()); vals = list(results.values())
    colors = ["gray"] + ["steelblue"] * (len(names) - 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, vals, color=colors)
    ax.axvline(n, ls="--", color="green", label=f"all {n} items")
    ax.set_xlim(0, n + 0.3); ax.set_xlabel(f"items retrievable (top-k) / {n}")
    ax.set_title("Day 4: slot memory vs single vector — retrievable items after sequence")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slot_memory.png"), dpi=130)
    plt.close(fig)
    print(f"\nartifact: {os.path.join(OUT, 'slot_memory.png')}")


if __name__ == "__main__":
    main()
