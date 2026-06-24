# WorldField — Architecture

A cognitive architecture built on a **single shared latent space**. Every block
reads and writes the *same* representation; there is no separate symbolic layer.

## The pipeline

```
                           ┌──────────────────────────────┐
   image ───►  ┌────────┐ │                              │
                │ encode │─┤      SHARED LATENT SPACE      │   one space,
   text  ───►  │  (text/ │ │   (128-dim L2-normalized,    │   every block
                │ image/  │ │    projected to same space)  │   lives here
   video ───►  │  video) │ │                              │
                └────────┘ └──────────────┬───────────────┘
                   PERCEPTION              │
                                           ▼
                             ┌──────────────────────────┐
                             │      FRAGMENT STORE       │   persistent vector
                             │  (ChromaDB: AI in latent │   DB; every input
                             │   vectors + metadata)     │   is a fragment
                             └─────────────┬─────────────┘
                                           │
                   ┌───────────────────────┼───────────────────────┐
                   ▼                       ▼                        ▼
         ┌───────────────────┐  ┌────────────────────┐  ┌────────────────────┐
         │    SLOT MEMORY    │  │     RETRIEVAL      │  │ GRAPH WORLD MODEL  │
         │  K slots, route   │  │  ChromaDB cosine,  │  │ PMI/lift edges     │
         │  by similarity,   │  │  top-k by score    │  │ over fragment ids  │
         │  merge / evict    │  │                    │  │ (co-occurrence)    │
         └─────────┬─────────┘  └─────────┬──────────┘  └─────────┬──────────┘
                   │                      │                       │
                   └──────────────────────┼───────────────────────┘
                                          ▼
                             ┌──────────────────────────┐
                             │      CONCEPT MEMORY       │
                             │  temporal decay,          │
                             │  confidence, uncertainty, │
                             │  hierarchy (parent/child) │
                             └─────────────┬─────────────┘
                                           ▼
                             ┌──────────────────────────┐
                             │       REASONING           │
                             │  spreading activation     │
                             │  over the fragment graph  │
                             │  (propagate / diffuse)    │
                             └─────────────┬─────────────┘
                                           ▼
                             ┌──────────────────────────┐
                             │       REFINEMENT          │
                             │  seed→propagate→re-seed→… │
                             │  damped, until converged  │
                             └─────────────┬─────────────┘
                                           ▼
                                        OUTPUT
                               (concept(s), confidence,
                                uncertainty, graded belief)
```

## Core blocks

| Block | What it does | Mechanism | Status |
|-------|--------------|-----------|--------|
| **Perception** | Encode text, image, video into shared space | SentenceTransformer (text), CNN (image), frame-sampling + CNN (video) | ✅ Text + Image, 🚧 Video |
| **Fragment Store** | Persistent vector storage | ChromaDB with cosine similarity | ✅ |
| **Retrieval** | Find fragments near a query | ChromaDB top-k search | ✅ |
| **Slot Memory** | Hold multiple concepts persistently | K slots, route by sim, merge ≥ threshold, LRU evict | ✅ |
| **Concept Memory** | Track concepts with confidence, uncertainty, hierarchy | Temporal decay, frequency-based confidence, parent-child structure | ✅ |
| **Graph World Model** | Learn relations between fragments | PMI/lift edge formation from co-occurrence | ✅ |
| **Reasoning** | Recover associates via graph | Row-stochastic spreading activation | ✅ |
| **Refinement** | Self-correct via damped iteration | EMA damping, propagate→re-seed loop | ✅ |
| **CLI** | Interactive chat + dashboard | rich Layout + prompt_toolkit input | ✅ |
| **Persistence** | Full state save/load | ChromaDB + JSON for slots/graph/concepts | ✅ |

## Design constraints

1. **Concepts are not points; events are not averages.** Slot memory exists because
   a single state vector collapses many concepts into one. Same rule governs edge
   formation: wire the union, never the averaged query.

2. **Cognition is a damped dynamical system, not a one-shot function.**
   Single-shot propagation is inert to noise. The loop needs EMA damping to
   converge instead of oscillate.

3. **Edges live on fragments, not on latent regions.** Two tokens may be nearby
   in latent space yet share no learned edges. The world-model graph is
   fragment-specific.

4. **Count association at the grain the world generates.** Edge formation must
   tally co-occurrence at concept/cluster granularity, not raw-fragment
   granularity.

## New in this version

- **Unified `worldfield` package** — pip-installable, all modules under one namespace
- **Sentence transformer text encoder** — replaces char-RNN with real language understanding
- **Video encoder** — frame sampling + per-frame CNN + temporal pooling
- **ChromaDB persistence** — fragments survive across sessions
- **Concept memory** — temporal decay, confidence tracking, uncertainty, hierarchy
- **CLI app** — interactive chat with live dashboard (rich TUI)
- **Continuous learning** — no epochs, each input updates state immediately

## What is *not* in the architecture yet

- **True causal direction discovery** — graph learns association, not causation
- **Calibrated uncertainty** — soft-mode tracking is approximate
- **Automatic hierarchy formation** — parent/child relationships are currently explicit
- **Real-world image/video generalization** — image encoder trained on synthetic shapes only

## The continuous loop

```
input → encode → store fragment → track concept → update slots → update graph → retrieve → respond
```

No batches. No epochs. Every input immediately updates:
1. Fragment store (adds the vector)
2. Concept memory (updates confidence/uncertainty/decay)
3. Slot memory (routes/merges/evicts)
4. Graph (records co-occurrence for PMI)

State is persisted after each step, so the system resumes exactly where it left off.
