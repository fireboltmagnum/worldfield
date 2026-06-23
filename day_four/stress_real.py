"""Day 4.5 Step B — confirm the slot-memory stress findings on REAL learned
latents (the 60-concept rich model), not synthetic geometry.

Same strict-vs-generous metric. The key extra honesty: the confusable concepts
here are REAL (blue/navy/royalblue/skyblue squares etc.), so routing faces
genuine learned similarity, not a similarity we hand-dialed.
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
from slot_memory import SlotMemory, _norm  # noqa: E402

CKPT = os.path.join(DAY1, "out", "worldfield_rich.pt")


def device():
    return torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")


def load(dev):
    if not os.path.exists(CKPT):
        sys.exit(f"No rich checkpoint at {CKPT}. Run day_one/train_rich.py first.")
    ck = torch.load(CKPT, map_location=dev)
    cfg = RichConfig()
    m = Worldfield(cfg, ck["vocab_size"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg, ck["class_names"]


def evidence_for(store, label, k=20):
    idx = np.where(store.labels == label)[0]
    pick = np.random.RandomState(label).choice(idx, size=min(k, len(idx)), replace=False)
    return store.vectors[pick].mean(0)


def score(store, mem, labels, k):
    slots = mem.active_slots()
    if slots.shape[0] == 0:
        return 0, 0
    top1, topk = [], []
    for s in slots:
        _, idx, _ = store.search(_norm(s).astype(np.float32)[None], k)
        row = store.labels[idx[0][idx[0] >= 0]]
        top1.append(int(row[0]) if len(row) else -1)
        topk.append(set(row.tolist()))
    t1 = set(top1)
    strict = sum(1 for l in labels if l in t1)
    generous = sum(1 for l in labels if any(l in tk for tk in topk))
    return strict, generous


def main():
    dev = device()
    model, cfg, names = load(dev)
    print(f"device {dev} | rich model: {len(names)} real concepts\n")

    # store of real image fragments
    ds = ShapesDataset(cfg, "train")
    store = FragmentStore(cfg.latent_dim, use_hnsw=False)
    vecs, labs = [], []
    for i in range(len(ds)):
        img, _, label = ds[i]
        with torch.no_grad():
            vecs.append(model.encode_image(img.unsqueeze(0).to(dev)).cpu().numpy()[0])
        labs.append(label)
    store.add(np.array(vecs, np.float32), np.array(labs, np.int64))

    name_to_idx = {n: i for i, n in enumerate(names)}

    def run(seq_names, n_slots, merge_threshold=0.6, k=20, eval_names=None):
        mem = SlotMemory(cfg.latent_dim, n_slots=n_slots, decay=0.5,
                         merge_threshold=merge_threshold)
        seq = [name_to_idx[n] for n in seq_names]
        for lab in seq:
            mem.update(evidence_for(store, lab, k))
        ev = [name_to_idx[n] for n in (eval_names or list(dict.fromkeys(seq_names)))]
        s, g = score(store, mem, ev, k)
        return s, g, len(ev), int(mem.used.sum())

    print("1. REAL CONFUSABLE concepts (the 4 blues, same shape)")
    blues = ["blue square", "navy square", "royalblue square", "skyblue square"]
    for mt in (0.6, 0.8, 0.9):
        s, g, n, used = run(blues, n_slots=8, merge_threshold=mt)
        print(f"   merge_threshold={mt} | slots used {used} for {n} blues | "
              f"strict {s}/{n}  generous {g}/{n}  gap {g-s}")

    print("\n2. OVER-CAPACITY on real concepts")
    some = names[:20]
    for nslots in (8, 16):
        s, g, n, used = run(some, n_slots=nslots)
        print(f"   {nslots} slots, 20 concepts | used {used} | "
              f"strict {s}/{n}  generous {g}/{n}  gap {g-s}")

    print("\n3. EVICTION: stream all 60 real concepts through 8 slots")
    s, g, n, used = run(names, n_slots=8)
    s_recent, g_recent, _, _ = run(names, n_slots=8, eval_names=names[-8:])
    print(f"   all 60 | strict {s}/60 generous {g}/60   ||   "
          f"last 8 concepts: strict {s_recent}/8 generous {g_recent}/8")

    print("\n4. RETURN-AFTER-ABSENCE on real concepts")
    # show 'teal pentagon', flood with 15 others, then bring it back
    target = "teal pentagon"
    flood = [n for n in names if n != target][:15]
    s_before, _, _, _ = run([target] + flood, n_slots=8, eval_names=[target])
    s_after, _, _, _ = run([target] + flood + [target], n_slots=8, eval_names=[target])
    print(f"   '{target}': after eviction {s_before}/1 -> after return {s_after}/1")

    print("\n=== READING (real latents) ===")
    print("Compare to Step A. If real confusable blues need a HIGHER merge_threshold")
    print("to split (because they're genuinely close), that's the honest cost of")
    print("the architecture: routing sensitivity must be tuned to real similarity.")


if __name__ == "__main__":
    main()
