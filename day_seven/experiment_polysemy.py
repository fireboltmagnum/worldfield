"""Day 7 — NATURAL ambiguity (polysemy): can ONE latent region hold two senses,
and can CONTEXT alone disambiguate them?

This is the deepest test of the whole 'shared latent space' thesis. Day 3 showed
a single point cannot hold many meanings; Day 5c showed injected 50/50 ambiguity
breaks single-shot. Polysemy is the real-world version: the SAME token ("bank")
means river-edge OR money-house, and only CONTEXT decides. The honest question:

    Does the same ambiguous latent region resolve to the RIGHT sense under
    context — or do the senses bleed (Day-3 collapse at the meaning layer)?

We do NOT inject the ambiguity. We use the confusable blues built on Day 1 for
exactly this purpose. Measured overlap (image-centroid cosine):
    royalblue <-> blue = 0.64   royalblue <-> navy = 0.60   blue <-> navy = 0.05
So 'royalblue square' is a genuine ambiguous token: a query for it retrieves a
real mix of the blue-sense and the navy-sense fragments. The substrate's own
geometry creates the ambiguity.

Setup (the polysemy analogy):
    AMBIGUOUS TOKEN : "royalblue square"   (the word "bank")
    sense A         : the BLUE neighborhood, wired to context  RIVER  (water-edge)
    sense B         : the NAVY neighborhood, wired to context  MONEY  (the institution)
    contexts        : two unrelated concepts (near-zero overlap with the blues and
                      each other) so context cannot be confused with the token.

Tests:
  1. CONTEXT STEERS: seed (ambiguous token + RIVER) -> must recover the BLUE sense
     and REJECT the NAVY sense; seed (token + MONEY) -> the reverse. Context is
     the ONLY thing that differs. This is the discriminator.
  2. NO-CONTEXT control: seed the ambiguous token ALONE. Honest outcomes:
       (a) holds BOTH senses ~50/50  -> uncertainty layer working at the meaning
           level (the Day-6.6 result, now over senses);
       (b) collapses to ONE sense    -> Day-3 bleed at the meaning layer.
     We report whichever happens. We do NOT fake (a).
  3. WRONG-CONTEXT control: seed (token + an UNRELATED context). It must NOT
     manufacture a confident sense — if it does, 'context steering' was an
     artifact of any extra activation, not real disambiguation.

Honesty guards:
  - ground-truth labels used ONLY for scoring, never for edges or seeding.
  - edges are wired from each sense's OWN fragments + its context (union, not
    averaged queries — the Day-5b lesson).
  - the ambiguous query is the SAME in tests 1a and 1b; only the context fragment
    differs. If recovery flips, it is the context doing the work.
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
from config_rich import RichConfig          # noqa: E402
from data import ShapesDataset              # noqa: E402
from model import Worldfield                # noqa: E402
from store import FragmentStore             # noqa: E402
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

    def frags(concept, k):
        pool = by_concept[idx[concept]]
        return RNG.choice(pool, size=min(k, len(pool)), replace=False)

    # ---- the polysemy roles ----------------------------------------------
    AMBIG = "royalblue square"      # the word "bank" (overlaps blue AND navy)
    SENSE_A_TOKEN = "blue square"   # river-sense surface form
    SENSE_B_TOKEN = "navy square"   # money-sense surface form
    CTX_RIVER = "green triangle"    # context that selects sense A (water-edge)
    CTX_MONEY = "red circle"        # context that selects sense B (institution)
    CTX_WRONG = "olive square"      # an unrelated context (must NOT steer)
    a_i, b_i = idx[SENSE_A_TOKEN], idx[SENSE_B_TOKEN]
    river_i, money_i = idx[CTX_RIVER], idx[CTX_MONEY]

    # confirm the ambiguous token really overlaps BOTH senses (and the contexts
    # do NOT) — print so the premise is visible, not assumed.
    def cen(name):
        ii = by_concept[idx[name]]; c = store.vectors[ii].mean(0)
        return c / (np.linalg.norm(c) + 1e-8)
    cA, cB, cAmb = cen(SENSE_A_TOKEN), cen(SENSE_B_TOKEN), cen(AMBIG)
    print("=== premise check: is the token genuinely ambiguous? ===")
    print(f"  ambig<->senseA(blue): {float(cAmb@cA):.2f}   "
          f"ambig<->senseB(navy): {float(cAmb@cB):.2f}   "
          f"senseA<->senseB: {float(cA@cB):.2f}")
    print(f"  ambig<->ctx_river: {float(cAmb@cen(CTX_RIVER)):.2f}   "
          f"ambig<->ctx_money: {float(cAmb@cen(CTX_MONEY)):.2f}   "
          f"(contexts must be ~0 = not confusable with the token)")

    # ---- learn the two senses (edges over fragment ids) -------------------
    # Each sense: its OWN fragments co-activate with its context. We wire the
    # union (NOT averaged queries — Day-5b lesson). Critically, the ambiguous
    # token's fragments are NEVER wired directly; the senses live on the blue and
    # navy neighborhoods that the ambiguous query happens to span.
    g = FragmentGraph(nF)
    for _ in range(20):
        ia, ir = frags(SENSE_A_TOKEN, 6), frags(CTX_RIVER, 6)   # blue ~ river
        g.observe(np.concatenate([ia, ir]), np.ones(12, np.float32))
        ib, im = frags(SENSE_B_TOKEN, 6), frags(CTX_MONEY, 6)   # navy ~ money
        g.observe(np.concatenate([ib, im]), np.ones(12, np.float32))

    def cscore(fs, top=200):
        per = {}
        for f in np.argsort(fs)[::-1][:top]:
            if fs[f] <= 0:
                break
            c = int(store.labels[f]); per[c] = per.get(c, 0.0) + fs[f]
        return per

    def sense_ratio(fs):
        """river-sense signal as a fraction of (river + money). The senses are
        identified by their CONTEXT associates (river vs money), recovered
        purely from propagation — the ambiguous token itself is the seed."""
        cs = cscore(fs)
        r, m = cs.get(river_i, 0.0), cs.get(money_i, 0.0)
        tot = r + m + 1e-9
        return r / tot, r, m

    def diffuse(state, hops=2, decay=0.5):
        """Mass-preserving spread (Day-6.6 operator), distinct from propagate
        which ZEROS the seed. Needed for the no-context case: the bare token's
        fragments only reach a sense THROUGH a context bridge, so zeroing the seed
        annihilates everything. diffuse keeps the seed's mass so we can ask the
        honest question — does the bare token hold BOTH senses, or one?"""
        Wc = g.W.tocsr()
        rs = np.asarray(Wc.sum(axis=1)).ravel()
        inv = np.zeros_like(rs); inv[rs > 0] = 1.0 / rs[rs > 0]
        from scipy.sparse import diags
        T = diags(inv).dot(Wc)
        x = state.copy(); out = state.copy()
        for _ in range(hops):
            x = T.T.dot(x) * decay
            out = out + x
        return out

    def run(token, context=None, hops=2, op="propagate"):
        """Seed the ambiguous TOKEN (+ optional context), spread, read which
        sense's context-associate comes back. Same token every time; only the
        context (and, for the bare-token control, the operator) changes."""
        seed_ids = list(frags(token, 6))
        if context is not None:
            seed_ids += list(frags(context, 6))
        seed_ids = np.array(seed_ids)
        vals = np.ones(len(seed_ids), np.float32)
        if op == "diffuse":
            state = np.zeros(nF, np.float32); state[seed_ids] = vals
            fs = diffuse(state, hops=hops)
        else:
            fs = g.propagate(seed_ids, vals, hops=hops)
        return sense_ratio(fs)

    print("\n=== TEST 1: does CONTEXT steer the ambiguous token? ===")
    print("    (river-sense fraction; ->1.0 = picked river/blue, ->0.0 = money/navy)")
    rA, _, _ = run(AMBIG, CTX_RIVER)
    rB, _, _ = run(AMBIG, CTX_MONEY)
    print(f"    token + RIVER context : river-sense = {rA:.2f}   "
          f"({'RIVER sense' if rA > 0.5 else 'money sense'})")
    print(f"    token + MONEY context : river-sense = {rB:.2f}   "
          f"({'RIVER sense' if rB > 0.5 else 'money sense'})")
    steers = rA > 0.6 and rB < 0.4          # flips the right way, decisively
    print(f"    -> context steering: {'YES' if steers else 'no'} "
          f"(spread {abs(rA - rB):.2f})")

    print("\n=== TEST 2: NO-context control (seed the bare ambiguous token) ===")
    print("    propagate ZEROS the seed (bare token reaches a sense only via a")
    print("    context bridge -> nothing survives). diffuse KEEPS seed mass, so it")
    print("    is the honest operator for 'does the token alone hold both senses?'")
    r0p, rrp, rmp = run(AMBIG, None, op="propagate")
    r0, rr, rm = run(AMBIG, None, op="diffuse")
    print(f"    propagate (zeros seed): river={rrp:.3f} money={rmp:.3f}  "
          f"-> {'dead (operator artifact, NOT a finding)' if rrp + rmp < 1e-6 else f'{r0p:.2f}'}")
    print(f"    diffuse  (keeps mass) : river={rr:.3f} money={rm:.3f}  river-sense={r0:.2f}")
    signal = (rr + rm) > 1e-6
    holds_both = signal and 0.30 <= r0 <= 0.70
    if not signal:
        msg = "no signal even with diffuse — token is not bridged to either sense"
    elif holds_both:
        msg = "HOLDS BOTH senses (~50/50): uncertainty at the meaning level"
    else:
        msg = "collapses to ONE sense (Day-3 bleed at the meaning level)"
    print(f"    -> {msg}")

    print("\n=== TEST 3: WRONG-context control (unrelated context must NOT steer) ===")
    rW, _, _ = run(AMBIG, CTX_WRONG)
    print(f"    token + UNRELATED context: river-sense = {rW:.2f}")
    no_artifact = abs(rW - r0) < 0.20       # unrelated context ~ no-context
    print(f"    -> unrelated context leaves the sense ~unchanged: "
          f"{'YES (good)' if no_artifact else 'NO — steering may be an artifact'}")

    # ---- verdict ----------------------------------------------------------
    print("\n=== VERDICT (Day 7 — natural ambiguity) ===")
    print(f"  context steers token to right sense: {steers} "
          f"(river-ctx {rA:.2f} vs money-ctx {rB:.2f})")
    print(f"  bare token: river-sense {r0:.2f} -> "
          f"{'both senses live' if holds_both else 'collapsed'}")
    print(f"  unrelated context does not fake a sense: {no_artifact}")
    if steers and no_artifact and holds_both:
        verdict = ("FULL PASS — ONE ambiguous region holds BOTH senses (bare token "
                   "~50/50) AND context resolves it correctly, with the artifact "
                   "control passing. Genuine substrate-level polysemy.")
    elif steers and no_artifact and not holds_both:
        verdict = ("QUALIFIED PASS — CONTEXT correctly disambiguates the ambiguous "
                   "token (artifact control passes), so context steering is real. "
                   "BUT the bare token does NOT independently hold both senses: it "
                   "recovers nothing. Mechanism is a CONTEXT BRIDGE, not the token "
                   "inheriting its neighbors' edges. Architectural lesson (echoing "
                   "Day 5b): the graph wires FRAGMENTS, not latent REGIONS — latent "
                   "similarity does not transfer learned edges to a nearby token. "
                   "Also note 1.00/0.00 is attractor-style, not graded.")
    elif steers and not no_artifact:
        verdict = ("SUSPICIOUS — context flips the sense, but an UNRELATED context "
                   "also moves it. The 'steering' may be any-extra-activation, not "
                   "genuine disambiguation. Not convinced.")
    else:
        verdict = ("FAIL — context does not reliably steer the ambiguous token to "
                   "the correct sense. The senses bleed: a single shared region "
                   "cannot hold polysemy under this mechanism. Honest negative — "
                   "Day-3 collapse reappears at the meaning layer.")
    print("\n" + verdict)


if __name__ == "__main__":
    main()
