# Worldfield — Findings

> Can a cognitive architecture be built on a single shared latent space?
>
> This document is the honest log of that question. Each "Day" tests **one**
> architectural claim. The goal was never to make things pass — it was to break
> the naive version of each capability and find the mechanism (if any) that makes
> it work. Where a result is an artifact, a plumbing fix, or holds only in a
> narrow regime, that is stated plainly.

## The thesis (what the arc actually shows)

A single shared latent substrate **can** support perception, retrieval,
persistent memory, associative reasoning, iterative self-correction, and graded
uncertainty — **but each capability requires a specific dynamical mechanism**
(slots, not one vector; damped iteration, not single-shot; distribution-updates,
not winner-take-all). The naive one-shot, single-point, winner-forcing versions
fail in **characteristic, diagnosable ways** — the canonical failure modes of
cognitive architectures, rediscovered from first principles.

## Recurring lessons (the through-line)

1. **Don't collapse a concept to a point.** (Day 3) A single state vector cannot
   hold many meanings; memory needs *slots*. The same lesson reappears at the
   *event* layer in Day 5b: averaging two concept queries lands on a third
   concept — a centroid is not its constituents.
2. **State needs inertia.** (Day 6) One-shot propagation is inert to seed noise;
   reasoning is a *dynamical system*, and an undamped loop oscillates. Damping
   (EMA) is what turns the loop into something that converges.
3. **The update rule decides certainty vs. calibration.** (Day 6.6) The same
   substrate, same evidence, same edges: a winner-forcing update is an
   *attractor* (collapses to 100/0); a distribution update is a *probabilistic
   reasoner* (preserves and tracks the evidence ratio). The architecture is more
   expressive than any single day suggested — the dynamics are a *choice*.
4. **The graph wires fragments, not regions.** (Day 5b, sharpened by Day 7)
   Latent *similarity* lets you retrieve a neighbor, but learned *edges* live on
   specific fragments and do **not** transfer to a similar token. A concept's
   relations are not inherited by what merely sits near it in latent space —
   meaning travels through wired co-activations (and, for an ambiguous token,
   through a context bridge), not through proximity alone.

---

## Scorecard

| Day | Claim under test | Verdict |
|-----|------------------|---------|
| 1   | A shared image+text latent space can be learned | ✅ Pass |
| 2   | Retrieval over fragments is robust to distractors/scale | ✅ Pass |
| 3   | A single persistent state can hold many concepts | ❌ Fail (the lesson) |
| 4   | Slot memory can hold many concepts at once | ✅ Pass |
| 4.5 | Slot memory survives stress (confusable / over-capacity / eviction / return) | ✅ Pass — strongest, with measured limits |
| 5a  | A co-activation graph recovers relations | ⚠️ Trivial (wrong object — ground-truth ids) |
| 5b  | Relations recover through the **real latent substrate** | ✅ Pass (clean seeds only) |
| 5c  | Reasoning survives **ambiguous/noisy seeds** | ❌ Fail (single-shot breaks at ≥40% contamination) |
| 6   | Iterative refinement self-corrects (overturns a wrong lead) | ⚠️ Pass on a tiny graph |
| 6.5 | Day 6 survives a **large, noisy, contradictory** graph | ✅ Pass (but saturates — a cliff, not a slope) |
| 6.6 | Worldfield can **represent uncertainty** (hold 60/40) | ✅ Pass — uncertainty is preservable; it's a *rule choice* |
| 7   | A single shared region can hold **two senses**, disambiguated by **context** | ⚠️ Qualified pass — context steering is real; the *token* doesn't hold both senses |
| 8   | Worldfield **discovers its own graph** from a raw stream (Hebbian + decay) | ❌ Fail on the core claim — it learns *frequency*, not *correctness* |
| 8b  | **PMI/lift** edge formation fixes Day 8 (reject above-chance) | ⚠️ Qualified pass — rejects spurious/contradictory, but narrow gate; confounds defeat it |
| 8c  | Is the PMI win robust to **representation granularity**? (stability audit) | ✅ Pass — narrow gate was a granularity artifact; separation is stable (~3–4×), label-free |

```
Perception → Retrieval → Memory → Reasoning → Self-correction → Uncertainty → Polysemy → Graph learning
    ✅           ✅          ✅         ✅             ✅          ✅            ⚠️    Hebbian❌ → PMI✅ assoc. (stable) → causal? (Day 9)
```

---

## Day 1 — Shared latent space

- **Claim:** Image and char-level text encoders can be trained into one shared,
  L2-normalized latent space via InfoNCE contrastive loss.
- **Experiment:** `day_one/train.py`, `day_one/train_rich.py`. Rich config: 12
  colors × 5 shapes = 60 concepts, 160 samples/class, 35 epochs. Checkpoints
  saved (`worldfield.pt`, `worldfield_rich.pt`) so later days reuse the encoders.
- **Result:** R@1 ≈ 0.99 on 60 concepts.
- **Failure:** None at this stage.
- **Fix:** N/A.
- **Proves:** A single contrastive latent space is learnable and cross-modal
  retrieval works at the concept level.
- **Does NOT prove:** Anything about memory, composition, or reasoning — only
  that the substrate exists and is well-separated. The dataset is synthetic
  shapes; near-duplicate colors (the blues) were added *deliberately* to create
  confusable concepts for later stress tests.

## Day 2 — Retrieval

- **Claim:** A fragment store over the latent space gives robust, fast retrieval.
- **Experiment:** `day_two/store.py` (FAISS `IndexFlatIP` exact / `IndexHNSWFlat`
  over L2-normalized vectors + parallel metadata + a top-k + similarity-threshold
  activation rule). `day_two/experiment.py`: semantic retrieval, distractor
  robustness, scale latency, sparsity.
- **Result:** precision@10 = 1.000 even with **95% distractors**.
- **Failure:** None.
- **Fix:** N/A.
- **Proves:** Retrieval is not the bottleneck; sparsity comes from the activation
  *rule* (threshold), not from FAISS approximation.
- **Does NOT prove:** That retrieval composes into memory or reasoning. It's a
  lookup, not a state.

## Day 3 — Single persistent state (the lesson)

- **Claim:** A single EMA-updated world-state vector can persistently hold
  several concepts at once (cat + room + sofa).
- **Experiment:** `day_three/world_state.py` (WorldStateEMA, plus
  WorldStateLastOnly as floor and WorldStateConcat as ceiling),
  `day_three/experiment.py`: capacity curve, decay sweep, honest margin metric.
- **Result:** **Split result.** By a generous margin metric, 5/6 concepts were
  "present"; by strict top-k retrievability, only **1/6** could actually be
  recovered.
- **Failure:** A single vector averages everything it absorbs; concepts
  interfere and only the most recent / strongest survives retrieval.
- **Fix:** None at the single-vector level — this is the *negative result* that
  motivates slots. **Lesson: don't collapse concepts to one point.**
- **Proves:** Persistent cognition cannot live in one averaged state vector.
- **Does NOT prove:** That memory is impossible — only that *this* representation
  is wrong. Directly motivates Day 4.

## Day 4 — Slot memory

- **Claim:** Memory split across K slots can hold many concepts simultaneously.
- **Experiment:** `day_four/slot_memory.py` (route by similarity; merge if
  best_sim ≥ merge_threshold, else claim a free slot, else LRU-evict).
  `day_four/experiment.py`: slot vs single-vector.
- **Result:** 8 slots → **6/6** concepts retrievable, vs. 1 vector → 1/6.
- **Failure (flagged immediately as suspicious):** 6 concepts in 5 slots scoring
  6/6 was a red flag — the metric used top_k=20, which is too generous.
- **Fix:** Re-examined with strict metrics in Day 4.5.
- **Proves:** Slots fix the Day-3 collapse — multiple concepts coexist.
- **Does NOT prove:** That it survives pressure (confusable concepts,
  over-capacity, eviction). That's Day 4.5.

## Day 4.5 — Slot memory under stress (the strongest result)

- **Claim:** Slot memory holds up under confusable concepts, over-capacity,
  eviction, and return-after-absence — measured by **strict** (top-1 own slot),
  not generous, metrics.
- **Experiment:** `day_four/stress_geometry.py` (synthetic controlled geometry)
  and `day_four/stress_real.py` (real 60-concept latents). Strict vs. generous
  scoring reported side by side.
- **Result:**
  - The strict/generous **inflation gap** was real — up to 4× — confirming the
    Day-4 suspicion.
  - 4 near-duplicate blues need merge_threshold ≈ 0.8 to split (3/4 merge at
    0.6).
  - 8 slots / 60 concepts → last 8 retained **7/8** under eviction.
  - Return-after-absence: a re-presented concept recovers (0 → 1).
- **Failure:** Effective capacity is **less than** slot count; routing threshold
  is sensitive; correlated concepts merge.
- **Fix:** Documented as known limitations in `day_one/README.md` — *run
  reasoning within capacity, tune merge_threshold to concept separation.*
- **Proves:** Slot memory is a genuine capability with characterized, honest
  limits — not a metric artifact.
- **Does NOT prove:** Unlimited capacity or automatic threshold selection.

## Day 5a — Co-activation graph over ground-truth ids

- **Claim:** A Hebbian co-activation graph ("fire together, wire together")
  recovers relations.
- **Experiment:** `day_five/coactivation.py` (integer-indexed graph),
  `day_five/experiment.py`.
- **Result:** Works — but trivially.
- **Failure:** It operates over **ground-truth concept ids**, not the latent
  substrate. Nobody tells a real system "cat = node 12." This was the *wrong
  object* — it skips the entire hard part.
- **Fix:** Relabeled honestly as "5a — graph propagation over ground-truth ids"
  with a scope note, and split the real test into 5b.
- **Proves:** Graph propagation mechanics are sound.
- **Does NOT prove:** Anything about reasoning on the actual latent space.

## Day 5b — Reasoning on the real latent substrate

- **Claim:** Relations recover through the **full** pipeline: latent query →
  retrieve fragments → form edges over **retrieved fragment ids** → propagate →
  retrieve again → score in latent space. Ground-truth labels used **only** for
  scoring, never for edges or seeding.
- **Experiment:** `day_five/experiment_latent.py` (`FragmentGraph`, scipy sparse
  `lil_matrix` over thousands of fragment ids; row-stochastic propagation that
  zeros the seed). Naive vs. guarded (similarity-floor) edge formation; control
  with no edges.
- **Result:** Recovered cat's associates (sofa ≈ 0.93, room ≈ 0.90), rejected
  the unrelated concept (bone), control with no edges = 0 recovered.
- **Failure (and the key finding):** First attempt recovered **nothing**. Root
  cause: building edges by **averaging** two concept queries (qa+qb) lands the
  midpoint on a *third* concept (red-circle ⊕ blue-square ≈ red-square). This was
  a genuine architectural finding, not just a bug — **the Day-3 lesson at the
  event layer: a centroid is not its constituents.**
- **Fix:** Activate each concept's fragments *separately* and wire the **union**
  (true co-activation), never the averaged query.
- **Proves:** Associative recovery works on the real substrate — the success
  belongs to the architecture, not a clean integer graph.
- **Does NOT prove:** Robustness to noisy/ambiguous seeds (tested clean seeds
  only). That's Day 5c.

## Day 5c — Reasoning under ambiguity (honest negative)

- **Claim:** Single-shot propagation survives a contaminated seed (cat query that
  also pulls in dog).
- **Experiment:** `day_five/experiment_ambiguity.py`: (A) controlled
  contamination sweep (seed = cat at weight 1−frac + dog at weight frac), and
  (B) realistic confusable query (a near-duplicate blue).
- **Result:** Recovery **breaks at ≥ 40% contamination**. More hops do **not**
  help — noise neither reliably cancels nor is managed.
- **Failure:** Single-shot propagation is *inert to seed noise*; once the wrong
  concept reaches plurality, propagation can't recover.
- **Fix:** None within single-shot. This negative result motivates Day 6 — the
  missing capability is *iteration*.
- **Proves:** A single propagation pass is not "reasoning" under ambiguity.
- **Does NOT prove:** That iteration can't fix it (it partly can — Day 6).

## Day 6 — Iterative refinement (self-correction)

- **Claim:** Looping (seed → propagate → re-seed → … → converge) can **overturn**
  a wrong initial plurality — genuine error-correction, not winner-take-all.
- **Experiment:** `day_six/experiment.py`. Fragment-level state with EMA damping
  (damp=0.6); convergence requires last-3-iters stable (tol=0.1). Explicit WTA
  baseline as the "fake version." Decisive test: dog 55% / cat 45% (wrong lead).
- **Result:** Refine flips to ~1.00 **stably**; WTA crawls to ~0.63,
  **unconverged**. The good-vs-fake discriminator passes: refine overturns the
  wrong lead, WTA cannot.
- **Failure:**
  - Collapse to 0.00 after iter 1 — re-seeding from arbitrary concept fragments
    hit unconnected nodes (only ~30 of 7680 fragments had edges).
  - Oscillation (0.45 → 1.00 → 0.12 → 1.00) — undamped full-replacement
    re-seeding.
- **Fix:** Re-seed at the **fragment level**, carry activation forward,
  restricted to connected fragments; add **EMA damping** + a convergence check.
  *Lesson: state needs inertia.*
- **Proves:** Iteration can supply the error-correction that single-shot lacked.
- **Does NOT prove:** That it isn't a tiny-graph attractor artifact — only ~30
  connected nodes. **This was the biggest red flag.** → Day 6.5.

## Day 6.5 — Refinement at scale (does the attractor survive?)

- **Claim:** Day 6's self-correction holds on a large, noisy, **contradictory**
  graph (hundreds of connected fragments; wrong edges baked in: cat–bone 10%,
  dog–room 15%).
- **Experiment:** `day_six/experiment_large.py`. Slope test: a *gradual* climb =
  integration; a single-step *cliff* = attractor artifact.
- **Result:** Flips to the correct associate and converges — but **snaps to
  1.00** (a cliff, not a gradual 0.45 → 0.58 → 0.71 slope).
- **Failure:** It saturates. Correct answer, but the dynamics are still
  attractor-like even at scale — it resolves rather than *integrates* in graded
  proportion.
- **Fix:** None needed for correctness; the saturation itself is the finding that
  sets up Day 6.6 (the winner-forcing update *is* the attractor).
- **Proves:** Self-correction is not merely a 30-node artifact — it survives
  scale and contradiction.
- **Does NOT prove:** That it can hold *graded* belief. It always resolves to a
  winner. → Day 6.6.

## Day 6.6 — Uncertainty representation (the capstone)

- **Claim:** Worldfield can *represent* uncertainty — hold 60/40 — rather than
  always collapsing to 100/0.
- **Experiment:** `day_six/experiment_uncertainty.py`. Graph bakes graded
  relational structure (cat co-occurs with room at p_room, with bone at p_bone).
  Compares the Day-6 **hard** update (keep top 50% + max-normalize → forces a
  peak) against a **soft** update (sum-normalized distribution over the connected
  backbone, no winner pruning), using a **new `diffuse()` operator** that keeps
  seed mass (distinct from Day-5b's `propagate`, which zeros the seed).
- **Result:**
  - Balanced evidence (50/50): hard → collapses to 1.00; **soft → holds ~0.50**.
  - Graded sweep, soft rule (room-signal vs. input room:bone ratio):
    `0.70 → 0.61 → 0.46 → 0.37 → 0.28` for inputs `0.70 / 0.60 / 0.50 / 0.40 /
    0.30` — **monotone, spread ≈ 0.40**. It tracks the evidence ratio.
  - Hard rule on the same inputs is **non-monotone and chaotic** (0.33, 0.64 for
    near-balanced inputs) — it falls into whichever basin, not the right
    proportion.
- **Failure / honesty:** This took **three debugging cycles**, and the soft rule
  initially output 0.00 for everything. Two real causes were found (not
  findings):
  1. Sum-norm diffusion into hundreds of fragments with a 1e-5 floor killed the
     signal; constraining to the backbone alone did **not** fix it.
  2. The real cause: `g.propagate` **zeros the seed** — fatal when the seed is
     the whole backbone (it annihilates a dense state).
  Fixed by adding the separate `diffuse()` operator that keeps seed mass.
- **Fix:** `diffuse()` — accumulate mass instead of zeroing the seed; soft update
  distributes (sum-norm) instead of peaking (max-norm).
- **Proves:** **Uncertainty collapse is not fundamental to the substrate — it's a
  property of the update rule.** The same graph is an attractor *or* a
  probabilistic reasoner depending on the dynamics. This is the most conceptually
  important result: the attractor-vs-probabilistic distinction is a *choice*.
- **Does NOT prove:**
  - That the soft mode is *calibrated* in any rigorous sense — tracking is
    approximate (0.50 → 0.46, slightly compressed/asymmetric), one RNG seed, one
    event count.
  - That `diffuse()` is the *same* mechanism used for reasoning — it is a **new
    operator**. "Worldfield can be both" is true, but it requires a
    purpose-built diffusion step, not a free switch. The honest framing:
    *retrieval-propagation and uncertainty-diffusion are distinct operations.*
  - That this is Bayesian belief updating (no priors, no normalization over a
    hypothesis space, no independence assumptions).

## Day 7 — Natural ambiguity (polysemy)

- **Claim:** A single ambiguous latent region can hold two senses of one token,
  and **context alone** can resolve it to the correct sense — the deepest test of
  the shared-latent-space thesis (Day 3 said a point can't hold many meanings;
  Day 5c said injected ambiguity breaks single-shot).
- **Experiment:** `day_seven/experiment_polysemy.py`. No injected ambiguity — uses
  the confusable blues built on Day 1. Measured overlap (image-centroid cosine):
  `royalblue↔blue = 0.64`, `royalblue↔navy = 0.60`, `blue↔navy = 0.05`. So
  `royalblue square` is a genuine ambiguous token ("bank"): the BLUE sense is
  wired to a *river* context, the NAVY sense to a *money* context. Three tests:
  (1) does context steer the same token to the right sense? (2) does the bare
  token hold both senses (~50/50)? (3) does an *unrelated* context fail to
  manufacture a sense (artifact control)?
- **Result:**
  - **Test 1 (context steering):** token + river → river-sense **1.00**; token +
    money → river-sense **0.00**. Same token, only context differs — it flips
    correctly.
  - **Test 3 (artifact control):** unrelated context leaves the sense unchanged —
    so the steering in Test 1 is *genuine disambiguation*, not "any extra
    activation tips it."
  - **Test 2 (bare token):** recovers **nothing** (river 0.000, money 0.000) —
    even with the mass-preserving `diffuse` operator, so it is **not** the
    seed-zeroing artifact.
- **Failure / honesty:** The script's auto-verdict printed a clean PASS claiming
  "one region carries two senses." That is **overclaimed** and was overruled:
  - The 1.00/0.00 split is an **attractor**, not graded disambiguation (Day-6.5
    cliff again — consistent with "the operator decides certainty").
  - The bare token holding nothing is **not** Day-3 bleed (the script's first
    label) and **not** an operator artifact (ruled out with `diffuse`). The real
    cause is structural: the `royalblue` *fragments* are distinct from the `blue`
    and `navy` *fragments* that carry the edges; latent **similarity** of
    centroids does not transfer **edges** to a nearby-but-distinct token.
- **Fix:** Added a `diffuse` operator to the no-context test to rule out the
  seed-zeroing artifact, and corrected the verdict logic to distinguish
  "holds both senses" from "context bridge only."
- **Proves:** Context-conditioned disambiguation of a genuinely ambiguous token
  **works**, with a proper artifact control. Meaning can be made contextual.
- **Does NOT prove:** That the token *itself* holds both senses. It doesn't —
  resolution runs through a **context bridge**, not through the ambiguous token
  inheriting its neighbors' relations. **Architectural lesson (echoing Day 5b):
  the co-activation graph wires fragments, not latent regions; latent similarity
  ≠ shared edges.** Also untested: graded (non-attractor) disambiguation, and
  true learned homonyms (one token string → two referents in perception).

## Day 8 — Unsupervised relation discovery (honest negative)

- **Claim:** Worldfield can **discover** its own relational graph from a raw,
  unlabeled observation stream — answering the strongest reviewer criticism that
  Days 5–7's relations were *authored*. Rule under test: **Hebbian with decay**
  (co-active fragments strengthen an edge; all edges decay each step, so unused
  edges are forgotten).
- **Experiment:** `day_eight/experiment_graph_learning.py`. A `DecayingHebbGraph`
  consumes a shuffled stream of observations (bags of co-fired fragment ids, no
  labels, no scripted "this is an A,B event"). The stream is deliberately
  hostile: true relations (frequent), **spurious** pairs (frequent-but-unrelated,
  lower rate), a **contradictory** relation (cat–bone, contradicts the true
  cat–{sofa,room} world, low rate), and pure-noise events. Labels used **only**
  for scoring. Three tests: (1) discovery, (2) commit×decay sweep, (3) the
  *killer* — rebuild Day-7 polysemy on the self-learned graph.
- **Result (after fixing weak controls — see below):**
  - True relations: mean edge weight **236**. Pure noise: **0.5** → true/noise
    **440×** (rare/random noise is crushed — decay works for *that*).
  - **Spurious: 107 (true/spur only 2.2×). Contradictory: 173 (true/contra only
    1.4×).** Structured wrong relations **survive** — they are nearly as strong as
    true ones.
  - Sweep: **0/4** decay settings cleanly separate true from contradictory.
    Higher decay does *not* help against the contradictory pair (stuck at 1.4×) —
    it recurs, so it gets refreshed exactly like a true relation.
  - `commit_rate` is a **pure scale knob** (ratio constant at 2.2 across
    0.5/1.0/2.0) — it cancels out of any ratio metric.
  - Killer test "passes" (polysemy 1.00/0.00) — but **only because that stream
    contained no contradictory sense-events**; it is the easy stream, so this
    "pass" is not evidence the rule works.
- **Failure / honesty:** My *first* run printed a clean PASS. It was wrong,
  caused by **weak controls**: (a) the contradictory pair was folded into
  "spurious" so it had no isolated readout; (b) the robustness sweep counted
  commit-rate columns as independent wins, but commit-rate cancels in a ratio —
  it was 3 decay results triple-counted as "9/12." Tightening the controls
  (isolating the contradictory pair, reporting per-knob sensitivity) flipped the
  verdict to FAIL. This is the entire value of the day.
- **Fix:** None — this is a **mechanism gap**, not a tuning problem. Decay
  punishes *infrequency*, not *wrongness*, so it cannot reject a recurring
  spurious/contradictory correlation.
- **Proves:** Hebbian + decay self-organizes against *random* noise (440×), and
  the experiment harness is now an honest discriminator for graph-learning rules.
- **Does NOT prove (the core failure):** That Worldfield discovers a *correct*
  world model. With this rule it is essentially a **frequency counter** —
  "fire together, wire together" learns whatever co-occurs often, true or false.
  **The reviewer's attack lands: graph FORMATION is the real gap.** The fix is a
  *commit rule that requires above-chance association* (PMI / lift), so a
  frequent-but-independent pair scores ~0 by construction. That is Day 8b.

## Day 8b — PMI/lift edge formation (the motivated fix)

- **Claim:** The Day-8 failure (learns frequency, not truth) is fixed by an
  **above-chance** commit rule. PMI asks "do A and B co-occur *more than
  chance*?" — `PMI = log(P(a,b)/(P(a)P(b)))` — so a frequent-but-independent pair
  scores ~0 by construction.
- **Experiment:** `day_eight/experiment_pmi_learning.py`. `PMIGraph` counts
  per-fragment and co-occurrence over the stream, then commits edges weighted by
  **PMI × co-count** (lift gated by support: `min_support`, `pmi_floor`).
  Identical hostile stream + controls as Day 8 (clean A/B). Pass bar = the bar
  Hebbian failed: true/contra > 5× AND true/spur > 5× AND polysemy survives.
  Test 4 adds the **confound** (C drives both A and B) to find PMI's ceiling.
- **Result:**
  - **true/spurious 2.2× → 11.1×; true/contradictory 1.4× → 12.7×.** PMI rejects
    exactly the pairs Hebbian could not. Per-true-relation weights uniform
    (98–123) — no weak edge hidden. Noise → 0.
  - **Polysemy survives** on the PMI-learned graph (1.00 / 0.00).
  - **Test 4 (the ceiling):** the confounded A–B edge is **0.94×** of a real
    edge — PMI wires a near-full-strength edge between two effects of a common
    cause. **Association is not causation.**
- **Failure / honesty:** Robustness is **narrow**. The support-gate sweep is
  clean **only at min_support=3** (support=1 is noisy at 2.1×; support≥5 collapses
  the whole graph to 0.0×). The script correctly reports "robust window: False."
  Root cause investigated: `frags()` sub-samples 6 of each concept's fragments
  per event, so a *specific* fragment-pair rarely recurs ≥5 times — the gate band
  is **partly an artifact of fragment sub-sampling**, not purely a property of
  PMI. A fairer test would accumulate co-counts at concept granularity or sample
  with repetition. So the win is real but the operating window is narrow and
  entangled with the sampling scheme.
- **Fix:** PMI × co-count with a support gate. The remaining brittleness is noted,
  not papered over.
- **Proves:** Edge **formation by association** is solved under proper controls —
  above-chance association rejects frequent-but-spurious and recurring-wrong
  relations that pure Hebbian frequency cannot, and the downstream capability
  (polysemy) transfers to the self-learned graph.
- **Does NOT prove:** (1) Robust learning across gate settings — only a narrow,
  partly-artifactual band works. **(Settled by Day 8c: the narrow band was a
  granularity artifact; the mechanism is stable.)** (2) **Causal** structure —
  PMI is fooled by a confound (0.94×). The system learns *what co-varies above
  chance*, not *what causes what*. → Day 9 is the clean, measured next step.
- **Number correction (per Day 8c):** the "11×" true/spurious figure was inflated
  by fragment-granularity noise at the knife-edge. The honest, representation-
  robust separation is **~3–4×** for the spurious pair (and near-total rejection
  for the contradictory pair). Smaller, but trustworthy.

## Day 8c — Stability audit (closing the methodological crack)

- **Claim:** The Day-8b PMI win is a real property of the mechanism, not an
  artifact of the fragment-sampling procedure. A skeptic's attack: "your support
  statistic only works at min_support=3; at 5 it zeroes — that's your sampling,
  not PMI."
- **Experiment:** `day_eight/experiment_stability_audit.py`. Identical stream and
  PMI rule; vary **only the representation** across four variants and sweep
  support [1,2,3,5,8,12], measuring whether the true/spurious separation is
  *invariant*: **A** fragment PMI, wide pool (160 — the Day-8b baseline); **B**
  fragment PMI, narrow pool (8 — aggressive reuse so pairs recur); **C**
  concept-level PMI (labels define units — the "cheating upper bound"); **D**
  fragment-**clustered** PMI (k-means, 60 clusters, **no labels** — the
  deployable version). Verdict tests *stability* (low spread across support),
  **not** the Day-8b 5× magnitude bar (wrong test for an audit).
- **Result:**
  - **A** (wide fragment pool): alive only 3/6, dies at support ≥ 5, true/spur
    [1.9, 3.7] — **unstable** (the fragility is here).
  - **B** (reuse): alive 6/6, true/spur **[3.2, 3.2]** — *perfectly invariant*.
    Reusing fragments removes the support sensitivity entirely.
  - **C** (concept-level): alive 6/6, true/spur [3.3, 4.3] — **stable**.
  - **D** (unsupervised clusters, **no labels**, 0.92 purity): alive 6/6,
    true/spur [3.1, 4.0] — **stable**. Contradictory pair rejected near-totally
    (ratio → ∞, weight ≈ 0) in C and D.
- **Failure / honesty:** My first run's auto-verdict printed "SURPRISE — PMI is
  fragile," which was **wrong** — a bug in the verdict logic, not a finding: it
  reused the 5× magnitude bar (B/C/D settle at ~3–4×, below it) and mishandled the
  contradictory ratio blowing up when contra-weight → 0 (which is the *best*
  outcome). Rewrote the verdict to test *invariance across support*, which is the
  audit's actual question. Also: this **corrects Day 8b's 11× downward to ~3–4×**
  — the larger figure was knife-edge noise.
- **Fix:** Diagnosis confirmed — count co-occurrence at **concept/cluster
  granularity**, not raw-fragment granularity; or reuse fragments so support is
  meaningful. The mechanism is sound; the original measurement unit was wrong.
- **Proves:** "PMI learns association" is **stable and understood** — the
  separation survives representation changes (fragment-reuse, concept-level, and
  *label-free* clustering) across a broad support window. The Day-8b narrow gate
  was a granularity artifact.
- **Does NOT prove:** Anything new about causality — that gap (confounds, Day-8b
  Test 4) stands, and is now the unambiguous next frontier because the
  association layer beneath it is verified solid.

---

## Open frontier (what's genuinely unanswered)

These are not toy variations — they're the questions that decide whether
Worldfield is a real architecture or a well-instrumented demo.

1. ~~**Can it learn the graph automatically?**~~ **(Days 8 / 8b — answered.)**
   Hebbian+decay learns frequency, not truth (Day 8 ❌). **PMI/lift fixes it**
   (Day 8b ⚠️): above-chance association rejects spurious/contradictory pairs
   (11×/13×) and polysemy survives — but the support gate is narrow and partly a
   sampling artifact. **Still open:** robust gate-free learning, and the deeper
   gap below (causation).
2. **Can it learn CAUSAL structure, not just association?** **(New, from Day 8b
   Test 4.)** PMI wires a confounded A–B edge at 0.94× of a real edge — two
   effects of a common cause look identical to a direct relation. A real world
   model needs *prediction / intervention / causal-consistency*, not co-variation.
   This is the next frontier (Day 9), now motivated by a measured failure.
2. ~~**Can it handle natural ambiguity?**~~ **(Day 7 — partially answered.)**
   Context *can* disambiguate a genuinely ambiguous token to the right sense
   (with an artifact control). But the token does not *itself* hold both senses;
   resolution runs through a context bridge because edges wire fragments, not
   regions. **Still open:** *graded* (non-attractor) disambiguation, and a true
   learned homonym (one token string → two perceptual referents) so the
   ambiguity lives in perception, not just in fragment proximity.
3. **Is uncertainty preservation calibrated and robust?** Sweep RNG seeds, event
   counts, graph sizes; quantify calibration error; test whether soft-mode
   tracking holds at scale.
4. **Unify the operators.** `propagate` (zeros seed, for one-shot recovery) and
   `diffuse` (keeps mass, for uncertainty) are two mechanisms. Is there one
   principled update rule with the winner-forcing-vs-distribution behavior as a
   single tunable parameter (a temperature)?
5. **Known plumbing leaks to close:** Day 6 clean-seed run drifts to 0.00 by
   iter 5; threshold/merge selection is manual; backbone-restriction is a
   workaround for sparse connectivity, not a designed mechanism.

## What I would not claim

- That any single day, alone, demonstrates "reasoning." The arc does; individual
  days demonstrate components and their failure modes.
- That synthetic shapes generalize to real perceptual data — untested.
- That the calibrated/uncertainty behavior is anything but emergent from a
  hand-chosen operator on one graph. It's a strong *existence proof*, not a
  characterization.
