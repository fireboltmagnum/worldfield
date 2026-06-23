# Worldfield — Architecture

A cognitive architecture built on a **single shared latent space**. Every block
reads and writes the *same* representation; there is no separate symbolic layer.
This document is the map; [FINDINGS.md](FINDINGS.md) is the experimental evidence
behind each block.

## The pipeline

```
                          ┌──────────────────────────────┐
   image ───►  ┌────────┐ │                              │
               │ encode │─┤      SHARED LATENT SPACE      │   one space,
   text  ───►  │  (CLIP-│ │   (L2-normalized vectors,    │   every block
               │  style)│ │    InfoNCE-contrastive)      │   lives here
               └────────┘ └──────────────┬───────────────┘
                  PERCEPTION              │
                                          ▼
                            ┌──────────────────────────┐
                            │     LATENT FRAGMENTS      │   atomic units of
                            │  (per-observation vectors │   experience; NOT
                            │   + label/strength/used)  │   "concepts"
                            └─────────────┬─────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                        ▼
        ┌───────────────────┐  ┌────────────────────┐  ┌────────────────────┐
        │    SLOT MEMORY    │  │     RETRIEVAL      │  │ GRAPH WORLD MODEL  │
        │  K slots, route   │  │  FAISS cosine,     │  │ co-activation edges│
        │  by similarity,   │  │  top-k + sim-      │  │ over FRAGMENT ids  │
        │  merge / evict    │  │  threshold rule    │  │ (Hebbian: fire     │
        │  (no single point)│  │                    │  │  together → wire)  │
        └─────────┬─────────┘  └─────────┬──────────┘  └─────────┬──────────┘
                  │  holds many          │ finds                 │ relates
                  │  concepts at once    │ fragments             │ fragments
                  └──────────────────────┼───────────────────────┘
                                         ▼
                            ┌──────────────────────────┐
                            │        REASONING          │   seed → propagate
                            │  spreading activation     │   over edges → read
                            │  over the fragment graph  │   back associates
                            └─────────────┬─────────────┘
                                          ▼
                            ┌──────────────────────────┐
                            │       REFINEMENT          │   LOOP with EMA
                            │  seed→propagate→re-seed→… │   damping; can
                            │  damped, until converged  │   OVERTURN a wrong
                            │                           │   lead (not WTA)
                            └─────────────┬─────────────┘
                                          ▼
                            ┌──────────────────────────┐
                            │     UNCERTAINTY LAYER     │   update-rule choice:
                            │  hard (max-norm)=attractor│   hard → 100/0
                            │  soft (sum-norm,diffuse)  │   soft → tracks 60/40
                            │  = probabilistic reasoner │
                            └─────────────┬─────────────┘
                                          ▼
                                       OUTPUT
                              (recovered concept(s) +
                               graded belief, in latent space)
```

## Blocks, and the experiment that validates each

| Block | What it does | Mechanism | Validated by | Status |
|-------|--------------|-----------|--------------|--------|
| **Perception** | Encode image & text into one space | Char-level + image encoders, InfoNCE | Day 1 — R@1 ≈ 0.99 | ✅ |
| **Latent Fragments** | Store per-observation vectors as atomic units | `FragmentStore` (vec + label/strength/last_used) | Day 2 | ✅ |
| **Retrieval** | Find fragments near a query | FAISS `IndexFlatIP` + top-k + sim-threshold activation | Day 2 — p@10 = 1.000 at 95% distractors | ✅ |
| **Slot Memory** | Hold *many* concepts persistently | K slots, route by sim, merge ≥ threshold, LRU evict | Day 4 / **Day 4.5** (strict-metric stress) | ✅ |
| **Graph World Model** | Learn relations between fragments | Hebbian co-activation edges over fragment ids (sparse) | Day 5a (mechanics) / **Day 5b** (real substrate) | ✅ |
| **Reasoning** | Recover an associate never directly queried | Row-stochastic spreading activation, score in latent space | Day 5b ✅ (clean) / Day 5c ❌ (ambiguous) | ⚠️ |
| **Refinement** | Self-correct: overturn a wrong lead | Damped (EMA) propagate→re-seed loop, convergence check | Day 6 / **Day 6.5** (scale + contradiction) | ✅ |
| **Uncertainty Layer** | Represent graded belief, not just a winner | Update-rule switch: max-norm (attractor) vs. sum-norm `diffuse` (probabilistic) | **Day 6.6** | ✅ |
| **Disambiguation** | Resolve an ambiguous token by context | Context-bridge: seed token + context, propagate, read the steered sense | **Day 7** | ⚠️ context-only |
| **Graph learning** | Discover edges from raw experience | Hebbian+decay → ❌ frequency; **PMI/lift** → above-chance association (count at concept/cluster granularity) | **Day 8 / 8b / 8c** | ✅ assoc. (stable, ~3–4×); ❌ causal |

## Two non-obvious design constraints (learned the hard way)

These are the parts that make the architecture *work* and were not obvious up
front — each was discovered by a failure (see FINDINGS for the breaks).

1. **Concepts are not points; events are not averages.** Slot memory exists
   because a single state vector collapses many concepts into one (Day 3). The
   *same* lesson governs edge formation: to wire an event with concepts A and B,
   activate each separately and wire the **union** — never the averaged query,
   whose midpoint lands on a *third* concept (Day 5b).

2. **Cognition is a damped dynamical system, not a one-shot function.**
   Single-shot propagation is inert to noise (Day 5c). The loop needs **EMA
   damping** to converge instead of oscillate (Day 6). And the *update rule*
   inside the loop is a first-class design knob: max-normalize forces an
   attractor; sum-normalize + a mass-preserving `diffuse` step yields a
   probabilistic reasoner (Day 6.6).

3. **Edges live on fragments, not on latent regions.** Two tokens can be highly
   similar in latent space yet share *no* learned edges — relations do not
   transfer by proximity (Day 5b, Day 7). An ambiguous token is resolved by a
   **context bridge** to a wired sense, not by inheriting its neighbors'
   relations. Retrieval is regional; the world-model graph is fragment-specific.

4. **Count association at the grain the world generates.** Edge *formation* must
   tally co-occurrence at concept/cluster granularity, not raw-fragment
   granularity (Day 8c). The world generates concept-level relations; a specific
   fragment-pair almost never recurs, so a support gate over fragment-pairs is a
   knife-edge artifact. Clustering fragments into concept-like units first
   (unsupervised, ~0.92 purity) makes the learned graph stable and support-
   invariant. *Edges live on fragments (constraint 3), but their statistics must
   be pooled over concept-grain units.*

## Two distinct propagation operators (do not conflate)

| Operator | Seed handling | Use | Where |
|----------|---------------|-----|-------|
| `propagate` | **zeros the seed** at the end | one-shot *recovery* (find the associate, not the query) | Day 5b/5c/6 |
| `diffuse` | **keeps seed mass**, accumulates | iterative *uncertainty* passing (a dense distribution must survive) | Day 6.6 |

Unifying these into one rule (with the attractor↔probabilistic behavior as a
single temperature parameter) is an open frontier item.

## What is *not* in the architecture yet

- **Causal structure** — the graph learner (Day 8b/8c) captures *above-chance
  association* robustly, but cannot distinguish a direct relation from two effects
  of a common cause (confounded edge wired at 0.94× of a real one). Learning *what
  causes what*, not just *what co-varies*, is the next frontier (Day 9).
- **Calibrated uncertainty** — soft-mode tracking (Day 6.6) is approximate and
  tested on one graph/seed; no calibration guarantee.
- **Unified propagation operator** — `propagate` (zeros seed) and `diffuse`
  (keeps mass) are two mechanisms; a single temperature-controlled rule would be
  cleaner.

**Resolved since the first draft:** automatic graph learning (Day 8b/8c — by
association, stable across representation granularity) and natural-ambiguity
disambiguation (Day 7 — by context bridge, with the caveat that the token itself
does not hold both senses).

See [FINDINGS.md](FINDINGS.md) for the full claim/break/fix/proves log and the
open-frontier list.
