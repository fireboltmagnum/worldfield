"""Day 8b — PMI/lift edge formation: does ABOVE-CHANCE association fix Day 8?

Day 8 (honest negative): Hebbian + decay learns FREQUENCY, not truth. A recurring
WRONG relation (cat~bone injected at low rate) survived almost as strongly as the
true relations (true/contradictory only 1.4x). Decay punishes infrequency, not
wrongness, so it cannot reject a frequent-but-spurious correlation.

The diagnosis points at the fix. PMI asks the right question:

    not "do A and B co-occur OFTEN?"   (Hebbian)
    but "do A and B co-occur MORE THAN CHANCE?"   (PMI / lift)

      PMI(a,b) = log( P(a,b) / (P(a) P(b)) )

A frequent-but-INDEPENDENT pair (sky+blue) has PMI ~ 0 by construction. A true
relation (cat+sofa) has PMI >> 0. So PMI should reject the spurious pair that
Hebbian could not.

We test on the IDENTICAL hostile stream + controls as Day 8 (a clean A/B):
  TRUE relations | SPURIOUS (frequent-but-unrelated) | CONTRADICTORY (recurring
  wrong) | NOISE. Pass bar = the bar Hebbian FAILED:
      true/contradictory > 5x   AND   true/spurious > 5x   AND   polysemy survives

Plus the BIGGER reviewer attack, baked in as Test 4: PMI still only sees
CO-OCCURRENCE, not CAUSAL structure. We build a CONFOUND — C drives both A and B,
so A and B co-occur with high PMI but are NOT directly related. If PMI wires a
strong A--B edge, we have shown its ceiling: association != causation. That
failure is not a bug — it is the motivation for Day 9 (prediction / causal
consistency).

Honesty notes baked in:
  - PMI over thousands of sparse fragments is noisy: a pair seen once can get
    explosive PMI from tiny counts. We gate PMI by SUPPORT (require a minimum
    co-count) and weight by co-count (this is 'lift gated by support', the
    standard fix). We report the gate so it is not hidden.
  - labels used ONLY for scoring, never for stream gen / edge formation / seeding.
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


class PMIGraph:
    """Edges from ABOVE-CHANCE association. We count, over a stream of events:
      n_i  = events in which fragment i fired
      n_ij = events in which i AND j both fired
      N    = total events
    Then PMI(i,j) = log( (n_ij N) / (n_i n_j) ). We commit an edge only when the
    pair has enough SUPPORT (n_ij >= min_support) and PMI > pmi_floor, and we
    weight the edge by  pmi * n_ij  (lift gated by support) so a one-off pair with
    huge PMI but tiny support cannot dominate.
    """
    def __init__(self, n, min_support=3, pmi_floor=0.0):
        self.n = n
        self.min_support = min_support
        self.pmi_floor = pmi_floor
        self.ni = np.zeros(n, dtype=np.float64)
        self.nij = {}        # (i,j) i<j -> co-count
        self.N = 0
        self.W = None

    def observe(self, frag_ids, acts=None):
        self.N += 1
        ids = np.unique(np.asarray(frag_ids))
        self.ni[ids] += 1
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                key = (int(ids[a]), int(ids[b]))
                self.nij[key] = self.nij.get(key, 0) + 1

    def finalize(self):
        """Turn counts into a PMI-weighted symmetric edge matrix."""
        W = lil_matrix((self.n, self.n), dtype=np.float32)
        N = max(self.N, 1)
        for (i, j), cij in self.nij.items():
            if cij < self.min_support:
                continue
            pij = cij / N
            pi, pj = self.ni[i] / N, self.ni[j] / N
            if pi <= 0 or pj <= 0:
                continue
            pmi = np.log(pij / (pi * pj) + 1e-12)
            if pmi <= self.pmi_floor:
                continue
            w = float(pmi * cij)         # lift gated by support
            W[i, j] += w; W[j, i] += w
        self.W = W
        return self

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

    # ---- identical world / stream as Day 8 (the A/B control) -------------
    TRUE = [("red circle", "blue square"),
            ("red circle", "green triangle"),
            ("yellow circle", "teal pentagon"),
            ("blue square", "purple diamond"),
            ("navy square", "olive square")]
    SPURIOUS = [("yellow circle", "green triangle")]
    CONTRA = [("red circle", "teal pentagon")]
    ALL_CONCEPTS = sorted({c for p in TRUE + SPURIOUS + CONTRA for c in p})

    def raw_stream(n_true=400, n_spur=40, n_contra=40, n_noise=120, fp=6):
        events = []
        for src, n in [(TRUE, n_true), (SPURIOUS, n_spur), (CONTRA, n_contra)]:
            for _ in range(n):
                a, b = src[RNG.randint(len(src))]
                events.append(np.concatenate([frags(a, fp), frags(b, fp)]))
        for _ in range(n_noise):
            events.append(RNG.choice(nF, size=2 * fp, replace=False))
        RNG.shuffle(events)
        return events

    def learn_pmi(stream, min_support=3, pmi_floor=0.0):
        g = PMIGraph(nF, min_support=min_support, pmi_floor=pmi_floor)
        for ids in stream:
            g.observe(ids)
        return g.finalize()

    def edge_concept_weight(g, ca, cb):
        Wc = g.W.tocsr()
        ia, ib = by_concept[idx[ca]], by_concept[idx[cb]]
        return float(Wc[np.ix_(ia, ib)].sum())

    # ============ TEST 1: discovery with PMI (vs Day-8 Hebbian) ============
    print("=== TEST 1: does PMI reject what Hebbian could not? ===")
    print("    (Day 8 Hebbian gave: true/spur=2.2x, true/contra=1.4x — FAILED)")
    stream = raw_stream()
    g = learn_pmi(stream, min_support=3, pmi_floor=0.0)
    true_w = np.mean([edge_concept_weight(g, a, b) for a, b in TRUE])
    spur_w = np.mean([edge_concept_weight(g, a, b) for a, b in SPURIOUS])
    contra_w = np.mean([edge_concept_weight(g, a, b) for a, b in CONTRA])
    rng2 = np.random.RandomState(1); noise_pairs = []
    pset = set(map(frozenset, [set(p) for p in TRUE + SPURIOUS + CONTRA]))
    while len(noise_pairs) < 8:
        a, b = rng2.choice(ALL_CONCEPTS, 2, replace=False)
        if frozenset({a, b}) not in pset:
            noise_pairs.append((a, b))
    noise_w = np.mean([edge_concept_weight(g, a, b) for a, b in noise_pairs])
    print(f"    (PMI gate: min_support=3, pmi_floor=0.0)")
    print(f"      TRUE relations      : {true_w:10.2f}")
    print(f"      SPURIOUS pairs      : {spur_w:10.2f}   "
          f"(true/spur   = {true_w/(spur_w+1e-9):6.1f}x)")
    print(f"      CONTRADICTORY pair  : {contra_w:10.2f}   "
          f"(true/contra = {true_w/(contra_w+1e-9):6.1f}x)")
    print(f"      NOISE pairs         : {noise_w:10.2f}   "
          f"(true/noise  = {true_w/(noise_w+1e-9):6.1f}x)")
    print("    per-true-relation weight:")
    for a, b in TRUE:
        print(f"      {a:16s} -- {b:16s}: {edge_concept_weight(g, a, b):9.2f}")
    rej_spur = true_w > 5 * max(spur_w, 1e-9)
    rej_contra = true_w > 5 * max(contra_w, 1e-9)
    discovered = rej_spur and rej_contra

    # ============ TEST 2: robustness to the support gate ============
    print("\n=== TEST 2: robustness across the support gate ===")
    print(f"    {'min_support':>12} | {'true/spur':>10} {'true/contra':>12} {'true/noise':>11}")
    wins = 0; gates = (1, 3, 5, 10)
    for ms in gates:
        gg = learn_pmi(stream, min_support=ms, pmi_floor=0.0)
        tw = np.mean([edge_concept_weight(gg, a, b) for a, b in TRUE])
        sw = np.mean([edge_concept_weight(gg, a, b) for a, b in SPURIOUS])
        cw = np.mean([edge_concept_weight(gg, a, b) for a, b in CONTRA])
        nw = np.mean([edge_concept_weight(gg, a, b) for a, b in noise_pairs])
        ok = tw > 5 * max(sw, 1e-9) and tw > 5 * max(cw, 1e-9)
        wins += int(ok)
        print(f"    {ms:>12d} | {tw/(sw+1e-9):>10.1f} {tw/(cw+1e-9):>12.1f} "
              f"{tw/(nw+1e-9):>11.1f}  {'clean' if ok else ''}")
    window = wins >= len(gates) // 2
    print(f"    -> {wins}/{len(gates)} support gates clean (real operating window: {window})")

    # ============ TEST 3: killer — polysemy on the PMI-learned graph ============
    print("\n=== TEST 3 (killer): polysemy on a PMI self-learned graph ===")
    AMBIG = "royalblue square"
    SENSE_A, SENSE_B = "blue square", "navy square"
    CTX_RIVER, CTX_MONEY = "green triangle", "red circle"
    river_i, money_i = idx[CTX_RIVER], idx[CTX_MONEY]
    poly_true = [(SENSE_A, CTX_RIVER), (SENSE_B, CTX_MONEY)]
    poly = []
    for _ in range(400):
        a, b = poly_true[RNG.randint(2)]
        poly.append(np.concatenate([frags(a, 6), frags(b, 6)]))
    for _ in range(120):
        poly.append(RNG.choice(nF, size=12, replace=False))
    RNG.shuffle(poly)
    gp = learn_pmi(poly, min_support=3)

    def sense_ratio(fs):
        per = {}
        for f in np.argsort(fs)[::-1][:200]:
            if fs[f] <= 0:
                break
            c = int(store.labels[f]); per[c] = per.get(c, 0.0) + fs[f]
        r, m = per.get(river_i, 0.0), per.get(money_i, 0.0)
        return r / (r + m + 1e-9)

    def disambig(context):
        seed = np.concatenate([frags(AMBIG, 6), frags(context, 6)])
        return sense_ratio(gp.propagate(seed, np.ones(12, np.float32), hops=2))

    rA, rB = disambig(CTX_RIVER), disambig(CTX_MONEY)
    print(f"    token + RIVER ctx: river-sense = {rA:.2f}")
    print(f"    token + MONEY ctx: river-sense = {rB:.2f}")
    poly_survives = rA > 0.6 and rB < 0.4
    print(f"    -> polysemy on PMI graph: {'SURVIVES' if poly_survives else 'BREAKS'}")

    # ============ TEST 4: the CEILING — association is not causation ============
    print("\n=== TEST 4 (the next attack): can PMI tell CAUSE from CONFOUND? ===")
    print("    C drives BOTH A and B. A,B co-occur (high PMI) but are NOT directly")
    print("    related. If PMI wires a STRONG A--B edge, that is its ceiling.")
    A, B, C = "magenta diamond", "skyblue pentagon", "crimson square"
    conf = []
    for _ in range(400):                       # C present -> A and B both appear
        conf.append(np.concatenate([frags(C, 6), frags(A, 6), frags(B, 6)]))
    for _ in range(120):
        conf.append(RNG.choice(nF, size=12, replace=False))
    RNG.shuffle(conf)
    gc = learn_pmi(conf, min_support=3)
    ac = edge_concept_weight(gc, A, C)
    bc = edge_concept_weight(gc, B, C)
    ab = edge_concept_weight(gc, A, B)         # the confounded (non-causal) edge
    print(f"    A--C (real)    : {ac:8.2f}")
    print(f"    B--C (real)    : {bc:8.2f}")
    print(f"    A--B (confound): {ab:8.2f}   "
          f"(fraction of a real edge: {ab/(max(ac,bc)+1e-9):.2f})")
    confounded = ab > 0.5 * max(ac, bc)        # PMI fooled by the confound
    print(f"    -> PMI {'IS fooled by the confound (association != causation)' if confounded else 'resists the confound'}")

    # ============ VERDICT ============
    print("\n=== VERDICT (Day 8b — PMI edge formation) ===")
    print(f"  rejects spurious : {rej_spur}  (true/spur   {true_w/(spur_w+1e-9):.1f}x; "
          f"Hebbian was 2.2x)")
    print(f"  rejects contra'y : {rej_contra}  (true/contra {true_w/(contra_w+1e-9):.1f}x; "
          f"Hebbian was 1.4x)")
    print(f"  robust window    : {window}")
    print(f"  polysemy survives: {poly_survives}")
    print(f"  confound ceiling : {'HIT (sets up Day 9)' if confounded else 'unexpectedly resisted'}")
    if discovered and poly_survives:
        print("\nPASS — PMI/lift edge formation FIXES the Day-8 failure: above-chance")
        print("association rejects the frequent-but-spurious AND the recurring")
        print("contradictory pair that Hebbian could not, and Day-7 polysemy survives")
        print("on the self-learned graph. Worldfield can construct a correct world")
        print("model from raw experience — by ASSOCIATION.")
        if confounded:
            print("\nBUT (Test 4) PMI is fooled by a CONFOUND: it cannot tell a direct")
            print("relation from two effects of a common cause. Association is not")
            print("causation. This is the clean, motivated next step -> Day 9.")
    elif discovered and not poly_survives:
        print("\nPARTIAL — PMI cleans the graph but polysemy does not survive; the")
        print("downstream capability does not transfer. See numbers above.")
    else:
        print("\nFAIL — even PMI does not separate true from spurious/contradictory")
        print("under these controls. Edge formation remains unsolved; see numbers.")


if __name__ == "__main__":
    main()
