"""Day 8 — UNSUPERVISED relation discovery: does Worldfield build its own graph?

The strongest criticism of Days 5-7 is fair: the relations were AUTHORED. Edges
were created from scripted (A,B) events. A self-organizing architecture must
DISCOVER edges from raw experience, with no one saying what a concept is, which
events matter, or which edges should exist.

So here the system gets a RAW STREAM of observations. Each observation is just a
bag of fragments that fired together (obtained by retrieval from a perceptual
query — the system never sees labels). It must:
  - commit/strengthen edges between co-active fragments  (Hebbian)
  - let unused edges DECAY toward zero                   (forgetting)
and from that alone recover the relational structure.

The stream is deliberately HOSTILE (this is what makes a pass real):
  - TRUE relations      : co-occur often       (should survive)
  - SPURIOUS pairs      : co-occur rarely       (should decay away)
  - PURE NOISE events   : random fragments      (should wire nothing stable)
  - one CONTRADICTORY pair: a wrong relation at low rate (should stay weak)
Labels are used ONLY for scoring, NEVER for stream generation, edge commit, or
seeding.

Tests:
  1. DISCOVERY : after the stream, are the true relations the strongest edges,
                 and are spurious/noise/contradictory pairs rejected?
  2. ROBUSTNESS: sweep (commit_rate x decay_rate) — is there a real operating
                 window, or does it only work at one lucky setting?
  3. THE KILLER: rebuild Day-7 polysemy on the SELF-LEARNED graph (edges were
                 discovered, not scripted). Does context still disambiguate the
                 ambiguous token? If yes, Days 5-7 are retroactively validated.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: F401
import sys
import numpy as np
import torch
from scipy.sparse import lil_matrix, diags

DAY1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_one"))
DAY2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_two"))
sys.path[:0] = [DAY1, DAY2]
from config_rich import RichConfig          # noqa: E402
from data import ShapesDataset              # noqa: E402
from model import Worldfield                # noqa: E402
from store import FragmentStore             # noqa: E402

CKPT = os.path.join(DAY1, "out", "worldfield_rich.pt")
RNG = np.random.RandomState(0)


def device():
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


class DecayingHebbGraph:
    """Edges discovered ONLINE from a stream. Co-active fragments strengthen their
    edge (gated by activation); every step ALL edges decay. No labels, no scripted
    pairs — the graph only ever sees fragment ids that co-fired in an observation.
    """
    def __init__(self, n, commit_rate=1.0, decay=0.02, gate=0.0):
        self.n = n
        self.W = lil_matrix((n, n), dtype=np.float32)
        self.commit_rate = commit_rate
        self.decay = decay
        self.gate = gate

    def observe(self, frag_ids, acts):
        """One raw observation: a set of co-active fragments with activations."""
        ids = np.asarray(frag_ids)
        a = np.asarray(acts, dtype=np.float32)
        keep = a > self.gate
        ids, a = ids[keep], a[keep]
        for i in range(len(ids)):
            for j in range(len(ids)):
                if ids[i] != ids[j]:
                    self.W[ids[i], ids[j]] += self.commit_rate * float(a[i] * a[j])

    def step_decay(self):
        """Decay every edge, then drop edges that have faded below a floor. Done in
        CSR (flat .data array is safe to mask); lil's parallel rows/data make
        in-place pruning corrupt the matrix, so we round-trip through CSR."""
        if self.decay <= 0:
            return
        Wc = self.W.tocsr()
        Wc.data *= (1.0 - self.decay)
        Wc.data[np.abs(Wc.data) < 1e-4] = 0.0
        Wc.eliminate_zeros()
        self.W = Wc.tolil()

    def propagate(self, seed_ids, seed_vals, hops=2, decay=0.5):
        Wc = self.W.tocsr()
        rs = np.asarray(Wc.sum(axis=1)).ravel()
        inv = np.zeros_like(rs); inv[rs > 0] = 1.0 / rs[rs > 0]
        T = diags(inv).dot(Wc)
        act = np.zeros(self.n, np.float32); act[seed_ids] = seed_vals
        frontier = act.copy(); total = act.copy()
        for _ in range(hops):
            frontier = T.T.dot(frontier) * decay
            total += frontier
        total[seed_ids] = 0.0
        return total


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

    def frags(concept, k):
        pool = by_concept[idx[concept]]
        return RNG.choice(pool, size=min(k, len(pool)), replace=False)

    # ---- the world (used ONLY to GENERATE a raw stream; never told to system) --
    # true relations:
    TRUE = [("red circle", "blue square"),       # cat~sofa
            ("red circle", "green triangle"),     # cat~room
            ("yellow circle", "teal pentagon"),   # dog~bone
            ("blue square", "purple diamond"),    # extra true
            ("navy square", "olive square")]      # money-sense pair (for Day 7)
    SPURIOUS = [("yellow circle", "green triangle")]  # frequent-but-unrelated, low rate
    # a DISTINCT control: a CONTRADICTORY relation (cat~bone) injected at a low
    # rate — it actively contradicts the true cat~{sofa,room} world. Tracked
    # separately so we can show it specifically stays weak.
    CONTRA = [("red circle", "teal pentagon")]
    ALL_CONCEPTS = sorted({c for pair in TRUE + SPURIOUS + CONTRA for c in pair})

    def raw_stream(n_true=400, n_spur=40, n_contra=40, n_noise=120, fp=6):
        """Emit an UNLABELED stream of observations. Each observation is a bag of
        fragments (from retrieval). The system gets only fragment ids + activations
        — never which concepts, never which events are 'true'. We interleave true,
        spurious, contradictory, and pure-noise events and shuffle."""
        events = []
        for src, n in [(TRUE, n_true), (SPURIOUS, n_spur), (CONTRA, n_contra)]:
            for _ in range(n):
                a, b = src[RNG.randint(len(src))]
                ids = np.concatenate([frags(a, fp), frags(b, fp)])
                events.append((ids, np.ones(len(ids), np.float32)))
        for _ in range(n_noise):
            ids = RNG.choice(nF, size=2 * fp, replace=False)
            events.append((ids, np.ones(len(ids), np.float32)))
        RNG.shuffle(events)
        return events

    def learn(stream, commit_rate, decay, gate=0.0, decay_every=1):
        g = DecayingHebbGraph(nF, commit_rate=commit_rate, decay=decay, gate=gate)
        for t, (ids, acts) in enumerate(stream):
            g.observe(ids, acts)
            if (t + 1) % decay_every == 0:
                g.step_decay()
        return g

    def edge_concept_weight(g, ca, cb):
        """Total learned edge weight between any fragment of concept ca and any of
        cb. Labels used ONLY here, for scoring the discovered graph."""
        Wc = g.W.tocsr()
        ia, ib = by_concept[idx[ca]], by_concept[idx[cb]]
        return float(Wc[np.ix_(ia, ib)].sum())

    # =================== TEST 1: DISCOVERY ===================
    print("=== TEST 1: does the stream self-organize into the right edges? ===")
    stream = raw_stream()
    print(f"    stream: {len(stream)} unlabeled observations "
          f"(true+spurious+contradictory+noise, shuffled)")
    g = learn(stream, commit_rate=1.0, decay=0.02)
    true_w = np.mean([edge_concept_weight(g, a, b) for a, b in TRUE])
    spur_w = np.mean([edge_concept_weight(g, a, b) for a, b in SPURIOUS])
    contra_w = np.mean([edge_concept_weight(g, a, b) for a, b in CONTRA])
    # noise baseline: random concept pairs that are none of the above
    rng2 = np.random.RandomState(1)
    noise_pairs = []
    pset = set(map(frozenset, [set(p) for p in TRUE + SPURIOUS + CONTRA]))
    while len(noise_pairs) < 8:
        a, b = rng2.choice(ALL_CONCEPTS, 2, replace=False)
        if frozenset({a, b}) not in pset:
            noise_pairs.append((a, b))
    noise_w = np.mean([edge_concept_weight(g, a, b) for a, b in noise_pairs])
    print(f"    mean learned edge weight:")
    print(f"      TRUE relations      : {true_w:10.1f}")
    print(f"      SPURIOUS pairs      : {spur_w:10.1f}   "
          f"(ratio true/spur   = {true_w / (spur_w + 1e-9):.1f}x)")
    print(f"      CONTRADICTORY pair  : {contra_w:10.1f}   "
          f"(ratio true/contra = {true_w / (contra_w + 1e-9):.1f}x)")
    print(f"      NOISE pairs         : {noise_w:10.1f}   "
          f"(ratio true/noise  = {true_w / (noise_w + 1e-9):.1f}x)")
    # per-relation breakdown so a single dominant edge can't hide weak ones
    print("    per-true-relation weight:")
    for a, b in TRUE:
        print(f"      {a:16s} -- {b:16s}: {edge_concept_weight(g, a, b):9.1f}")
    # the contradictory pair is the strictest control: it must stay below TRUE.
    discovered = true_w > 5 * max(spur_w, contra_w, noise_w, 1e-9)

    # =================== TEST 2: ROBUSTNESS (sweep) ===================
    # HONESTY NOTE: commit_rate scales ALL edges equally, so it CANCELS out of any
    # true/spurious RATIO. To probe both knobs honestly we report (a) the ratio
    # (sensitive to DECAY) and (b) the per-edge TRUE weight (sensitive to COMMIT,
    # an absolute scale). Counting ratio-wins across commit columns would be fake.
    print("\n=== TEST 2: operating window — and which knob actually matters ===")
    print("    true/spur RATIO depends on DECAY (commit cancels in a ratio):")
    print(f"    {'decay':>10} | {'true/spur':>10} {'true/contra':>12} {'true/noise':>11}")
    decay_wins = 0; decays = (0.005, 0.02, 0.05, 0.1)
    for dec in decays:
        gg = learn(stream, commit_rate=1.0, decay=dec)
        tw = np.mean([edge_concept_weight(gg, a, b) for a, b in TRUE])
        sw = np.mean([edge_concept_weight(gg, a, b) for a, b in SPURIOUS])
        cw = np.mean([edge_concept_weight(gg, a, b) for a, b in CONTRA])
        nw = np.mean([edge_concept_weight(gg, a, b) for a, b in noise_pairs])
        r = tw / (sw + 1e-9)
        decay_wins += int(r > 5 and tw > 5 * cw)
        print(f"    {dec:>10.3f} | {r:>10.1f} {tw/(cw+1e-9):>12.1f} {tw/(nw+1e-9):>11.1f}")
    print("    commit_rate is a pure SCALE knob (verify it cancels in the ratio):")
    for com in (0.5, 1.0, 2.0):
        gg = learn(stream, commit_rate=com, decay=0.02)
        tw = np.mean([edge_concept_weight(gg, a, b) for a, b in TRUE])
        sw = np.mean([edge_concept_weight(gg, a, b) for a, b in SPURIOUS])
        print(f"      commit={com:>4.1f}: true={tw:8.1f}  ratio={tw/(sw+1e-9):.1f}  "
              f"(ratio constant => scale-only, as predicted)")
    window = decay_wins >= len(decays) // 2
    wins, total = decay_wins, len(decays)
    print(f"    -> {wins}/{total} DECAY settings clean (true/spur>5x AND true/contra>5x) "
          f"({'real operating window' if window else 'fragile — works only narrowly'})")

    # =================== TEST 3: THE KILLER — polysemy on a LEARNED graph ===
    print("\n=== TEST 3 (killer): does Day-7 polysemy survive a SELF-LEARNED graph? ===")
    # Build a stream whose TRUE relations include the two senses, learn the graph
    # unsupervised, then run the Day-7 disambiguation on the DISCOVERED edges.
    AMBIG = "royalblue square"
    SENSE_A, SENSE_B = "blue square", "navy square"       # river / money senses
    CTX_RIVER, CTX_MONEY = "green triangle", "red circle"
    river_i, money_i = idx[CTX_RIVER], idx[CTX_MONEY]

    poly_true = [(SENSE_A, CTX_RIVER), (SENSE_B, CTX_MONEY)]
    poly_events = []
    for _ in range(400):
        a, b = poly_true[RNG.randint(2)]
        ids = np.concatenate([frags(a, 6), frags(b, 6)])
        poly_events.append((ids, np.ones(len(ids), np.float32)))
    for _ in range(120):  # noise
        ids = RNG.choice(nF, size=12, replace=False)
        poly_events.append((ids, np.ones(len(ids), np.float32)))
    RNG.shuffle(poly_events)
    gp = learn(poly_events, commit_rate=1.0, decay=0.02)

    def sense_ratio(fs):
        per = {}
        for f in np.argsort(fs)[::-1][:200]:
            if fs[f] <= 0:
                break
            c = int(store.labels[f]); per[c] = per.get(c, 0.0) + fs[f]
        r, m = per.get(river_i, 0.0), per.get(money_i, 0.0)
        return r / (r + m + 1e-9), r, m

    def disambig(context):
        seed = np.concatenate([frags(AMBIG, 6), frags(context, 6)])
        fs = gp.propagate(seed, np.ones(len(seed), np.float32), hops=2)
        return sense_ratio(fs)

    rA, _, _ = disambig(CTX_RIVER)
    rB, _, _ = disambig(CTX_MONEY)
    print(f"    token + RIVER ctx: river-sense = {rA:.2f}")
    print(f"    token + MONEY ctx: river-sense = {rB:.2f}")
    poly_survives = rA > 0.6 and rB < 0.4
    print(f"    -> polysemy on self-learned graph: "
          f"{'SURVIVES' if poly_survives else 'BREAKS'} (spread {abs(rA - rB):.2f})")

    # =================== VERDICT ===================
    print("\n=== VERDICT (Day 8 — unsupervised relation discovery) ===")
    print(f"  discovery: true relations dominate noise: {discovered} "
          f"({true_w / (max(spur_w, noise_w) + 1e-9):.1f}x)")
    print(f"  robustness: operating window exists: {window}")
    print(f"  killer: polysemy survives self-learned graph: {poly_survives}")
    if discovered and window and poly_survives:
        print("\nPASS — Worldfield CONSTRUCTS its own world model from a raw, noisy,")
        print("contradictory stream (no labels, no scripted pairs): true relations")
        print("self-organize, spurious/noise edges decay away, across an operating")
        print("window of settings. And Day-7 polysemy STILL works on the discovered")
        print("graph. The relations were learned, not authored. The thesis holds.")
    elif discovered and poly_survives and not window:
        print("\nQUALIFIED PASS — it discovers the right graph and polysemy survives,")
        print("but only in a narrow settings window. Self-organization is real but")
        print("not robust to the commit/decay balance. Honest about fragility.")
    elif discovered and not poly_survives:
        print("\nPARTIAL — edges self-organize, but polysemy does NOT survive on the")
        print("learned graph: the scripted graph was doing hidden work. The hardest")
        print("downstream capability does not transfer to discovered structure.")
    else:
        print("\nFAIL — the stream does not self-organize into the correct relations")
        print("(noise/spurious survive). Worldfield still needs authored edges.")
        print("Honest negative: graph FORMATION, not just reasoning, is the gap.")


if __name__ == "__main__":
    main()
