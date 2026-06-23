# WorldField – The Full Explanation

Alright, so here's what I was actually trying to do, in human terms.

Most AI systems are broken up into separate parts: one model for text, one for images, one for memory, maybe another for reasoning. They're like different silos. WorldField is me asking: what if they all shared the same underlying "meaning space"?

Like, imagine a student who could look at a picture of a red square, hear the words "red square," and remember seeing one before – and all three of those things would light up the same region of their brain. That's the vibe.

## What's a Latent Space, Actually?

A latent space is just a hidden number-space where meaning lives.

So when I say:
- a picture of a red square
- the text "red square"  
- a memory of a red square

All three should end up in roughly the same neighborhood of numbers. Not as English text—as vectors, which are just lists of floats. If two things mean the same thing, their vectors are close. If they're different, they're farther apart.

That's literally the core idea.

## Direct Multimodality

OK so "direct multimodality" just means: text and images go into the same space without being converted to one or the other first.

Right now WorldField only handles:
- text
- images

Images get turned into vectors by a small CNN. Text gets turned into vectors by a character-level RNN. They're different encoders because pixels and words are totally different things. But *after* encoding, both just become vectors in the same shared space.

```text
image -> CNN encoder -> shared latent vector
text  -> RNN encoder -> shared latent vector
```

After that, the system doesn't think "I'm in image mode" or "I'm in text mode." It just works with vectors. Same latent space. Same rules.

So this is actually important because it's not just "glue two models together." It's more like:

```text
photo of red square ----\
                        +--> both light up same region
"red square" text ------/
```

Not two separate things that happen to be connected. They're actually in the same space.

## The Full Pipeline (Simplified)

```text
text/image in
    |
    v
shared latent space (the one unified meaning-space)
    |
    v
pull out latent "fragments" (pieces of stored meaning)
    |
    +---> retrieval (find nearby fragments)
    +---> slot memory (keep multiple things active at once)
    +---> graph (which fragments relate to which?)
    |
    v
propagate activation through the graph
    |
    v
refine and settle (maybe over multiple steps)
    |
    v
either: one hard answer, or: soft graded belief
```

What each piece does:

- **Shared latent space**: the one common zone where everything lives
- **Fragments**: stored pieces of experience as vectors
- **Retrieval**: "what's nearby?" 
- **Slot memory**: "keep track of multiple concepts at the same time"
- **Graph**: "which things are related?"
- **Propagation**: "spread activation through the graph"
- **Refinement**: "let the state settle over time"
- **Uncertainty**: "hard collapse" vs "soft belief"

## About LSNN

You asked about LSNN. Yeah, LSNN is a real thing – it's not just me making stuff up.

LSNN = **Long Short-Term Memory Spiking Neural Network**

The main paper is Bellec et al., "Long short-term memory and learning-to-learn in networks of spiking neurons" (NeurIPS 2018). There's also an official repo if you want to see it: `IGITUGraz/LSNN-official`.

So what's an LSNN? It's a recurrent neural network made of spiking neurons. "Spiking" means the neurons fire discrete spike events, kind of like real neurons in your brain. The cool part is that LSNN neurons are *adaptive*—they change their excitability based on recent firing. So the network gets built-in memory just from how the neurons adapt over time.

The core ideas in LSNN:
- spiking neurons (discrete events)
- recurrence (feedback loops)
- sparse activity (not everything fires all the time)
- adaptation (neurons change over time)
- memory baked into neural dynamics

WorldField is **not** actually implementing LSNN. I'm not simulating spikes, membrane voltage, or any of that neuroscience. But I'm thinking in a similar vein:

| LSNN Concept | What WorldField Does Instead |
|---|---|
| Spiking neurons | Sparse fragment activation |
| Recurrent dynamics | Refinement loop (update state multiple times) |
| Neuron adaptation | Memory in slots + damping |
| Time-dependent computation | Multiple refinement steps matter |

So honest way to say it: WorldField isn't an LSNN, but it's aiming at the same gut feeling—sparse state, memory over time, computation through dynamics rather than static lookup.

## What's Actually Here Right Now

This repo isn't a product. It's more like a research lab notebook made of experiments.

Tech stack:
- Python
- PyTorch (for encoders)
- NumPy (for vectors and graph stuff)
- FAISS (for nearest-neighbor retrieval)
- Matplotlib (for plots)
- SciPy + scikit-learn (for analysis)

What it can currently test:
- Do image and text actually align in the same space?
- Can we retrieve the right stuff from memory?
- Does one vector collapse under multiple concepts?
- Can slot memory fix that?
- Can graphs store relationships?
- Does the system recover from wrong first guesses?
- Can it handle ambiguous input?
- Can it learn associations automatically?
- Can it start thinking about causality?

What it does **not** prove yet:
- That it'd work on real-world images
- Audio or video
- Actual causal direction
- Fragment-level causality
- Perfect uncertainty everywhere
- Text generation
- Lifelong learning with new concepts

That's all future stuff.

## Experiment Map

Each day is basically one question:

```
Day 1  → Can images and text actually share one space?
Day 2  → Can I retrieve the right thing from memory?
Day 3  → Can ONE vector hold multiple concepts? (spoiler: no)
Day 4  → Does slot memory fix it? (yes, kind of)
Day 5  → Can graphs learn relationships?
Day 6  → Can the system fix wrong first guesses?
Day 7  → Can context resolve ambiguous meanings?
Day 8  → Can the graph learn itself without me writing it?
Day 9  → Can it start thinking about causality?
```

## Day 1: Can Text and Images Actually Share One Space?

**What I tried:**

Made a bunch of synthetic colored shapes (red squares, blue circles, etc.). Also made matching text descriptions like "a red square." Then I trained two encoders—one for images, one for text—to push both the image and its matching text toward the same spot in the latent space using contrastive loss.

**What I got:**

- **R@1 ~= 0.99** on the test set (recall at rank 1—basically, the matching text is almost always the top hit)
- When I plotted the latent space, images clustered by color and shape, and text labels clustered the same way
- When I reconstructed images from the latent vectors, they still had the right shape and color

**Why it matters:**

This proved that direct multimodality actually works, at least in a toy world. Text and images can genuinely share one meaning-space. It's not just theory.

## Day 2: Retrieval from Memory

**The question:**

If I store a bunch of random latent vectors and then search for nearby ones, does retrieval actually work?

**What I tried:**

Stored a bunch of latent vectors as "fragments." Used FAISS (a fast nearest-neighbor library) to find the closest ones. Added a ton of decoy vectors to make it harder.

**What I got:**

- **Precision@10 = 1.0 even with 95% distractors**

Basically, retrieval stayed perfect even when the memory was full of noise.

**Why it matters:**

The whole architecture needs "sparse activation"—only wake up the relevant memories, not everything. Retrieval is how you do that. If it breaks, the whole thing breaks. But it doesn't—it's solid.

## Day 3: One Vector Can't Hold Multiple Concepts

**The question:**

What if I try to squish multiple concepts into a single latent vector? Can one vector be a world state?

**What I tried:**

Updated a single latent vector with multiple concept vectors one by one. Then tried to retrieve the original concepts from that single vector.

**What I got:**

Using loose metrics, it looked kind of okay. But with strict retrieval metrics? Only **1 out of 6** concepts actually came back.

**The lesson:**

One vector becomes a blurry average. It can't actually hold multiple distinct concepts. This is a hard failure. It's important because it tells you what *not* to do.

## Day 4: Slot Memory (The Fix)

**The question:**

So one vector fails. What if instead of one slot, I use multiple slots? Like a buffer with multiple memory slots that hold different concepts?

**What I tried:**

Made a memory with 8 slots. New concept goes to the closest slot (if it's close enough). If it's closer to something already in a slot, it merges. If nothing is close, it claims a new slot. If all slots are full and nothing's close, it kicks out the least-recently-used slot.

**What I got:**

- Clean test: **6/6 concepts recovered**
- But stress tests showed limits:
  - Can't actually hold 8 concepts (real capacity is lower)
  - Similar concepts sometimes accidentally merge
  - The merge threshold is really important to tune
  - If you overfill it, stuff gets evicted and lost

**The idea:**

Instead of this:
```
cat + sofa + room → one blurry vector (fails)
```

Do this:
```
slot 1 → cat
slot 2 → sofa
slot 3 → room
```

Much better, but not unlimited.

## Day 5: Graph Reasoning

**The question:**

OK, so I have fragments and slots. But how do concepts *relate*? If I've seen "cat" and "sofa" together before, should I expect them to be related?

**What I tried:**

Built a co-activation graph. When two fragments activate together, the edge between them gets stronger. Later, if I seed the graph with "cat," activation spreads through edges to find related things.

**What I got:**

- Graph mechanics worked
- Clean reasoning worked (e.g., "cat" → "meows" was solid)
- But averaging two queries failed badly. If I averaged "red circle" and "blue square," it'd land near "red square" (neither of the originals!)

**The hard lesson:**

Don't average two concepts and use the average as a query. It's a fake third concept. Instead, activate *both* fragments and propagate from both of them.

## Day 5b: When One Hop Fails

**The question:**

What if the input is noisy? Can one propagation step still find the right answer?

**What I tried:**

Mixed cat and dog signals (40% contamination, 60% signal) and asked the graph to find the associate.

**What I got:**

Started breaking around 40% contamination. More hops didn't really fix it.

**The lesson:**

One shot ain't enough. You need something that evolves over time.

## Day 6: Iterative Refinement

**The question:**

What if I let the system run for multiple steps? Can it correct a wrong first guess?

**What I tried:**

- Started with a wrong seed
- Propagated activation through the graph
- Re-seeded at the fragment level
- Carried state forward with exponential moving average (EMA) damping to prevent oscillation
- Did it all again

**What I got:**

Refinement could actually overturn the wrong lead and settle on the right answer. Winner-take-all didn't work as well.

**Visual:**

```
messy first state
      |
      v
propagate
      |
      v
re-seed + damping
      |
      v
propagate again
      |
      v
settled answer
```

**The idea:**

Reasoning is a *process*, not an instant jump. Let it evolve.

## Day 6.5 & 6.6: Handling Uncertainty

**Day 6.5:**

Ran refinement on a bigger, noisier graph. It still found the right answer but snapped to 1.0 confidence like an attractor.

**Day 6.6 – The real question:**

Can the system hold *uncertainty* instead of collapsing to certainty?

**What I tried:**

- Hard mode: standard update (delete old state, set new state)
- Soft mode: keep some of the old state, diffuse it (spread the mass around)

**What I got:**

- Hard mode → collapses to 1.0, picks a winner
- Soft mode → keeps graded belief like 60/40 or 50/50

**The insight:**

The latent space isn't *forcing* certainty. The *update rule* is. Use soft updates and you can keep uncertainty.

## Day 7: Context Resolves Ambiguity

**The question:**

What if a token is ambiguous? Like "bank" could mean money or a river. Can context steer which meaning gets activated?

**What I tried:**

Used confusable colors as stand-ins for ambiguous words. Put one meaning in one context, another meaning in another context.

**What I got:**

- Context steering worked – the right context lit up the right meaning
- Wrong context didn't fake it – it didn't force a false meaning
- But the bare ambiguous token itself didn't hold *both* meanings at once

**The catch:**

Context can steer, but the underlying graph edges live on *actual fragments*, not just on nearby latent regions. So disambiguation happens through the graph structure, not magic.

## Day 8: Can the System Learn Its Own Graph?

**The question:**

Instead of me writing the relationships by hand, can WorldField learn them automatically?

**Day 8a – Hebbian learning:**

*"Neurons that fire together, wire together."*

What I got:

- Random noise got crushed (good)
- But frequent wrong relations survived (bad)
- It learned *frequency*, not truth

**Day 8b – PMI (Pointwise Mutual Information):**

Only wire things that co-occur more than chance. It's like asking: "Is this pair appearing together more often than random luck would predict?"

What I got:

- Much better than Hebbian
- Rejected spurious + contradictory pairs
- Still got fooled by confounds (when a hidden cause makes two things seem related)

**Day 8c – Stability check:**

Tested whether the PMI result was stable.

What I got:

- Fragment-pair counting was brittle
- But concept/cluster-level counting was stable
- Unsupervised clusters got about **0.92 purity**

**The lesson:**

Association learning works better with PMI, but association ≠ causation. You can learn what goes together, but you can't yet distinguish real edges from confounded ones.

## Day 9: Concept-Level Causal Skeleton Recovery

**The question:**

Can the graph tell the difference between a real relationship and a *confound*?

**What's a confound?**

```
      C
     / \
    A   B
```

A and B look related because they both come from C. But there's no direct edge A–B.

**What I tried:**

- Tested conditional-independence screening
- Checked: single confounds, chains, colliders
- Later tested: pure conditional mutual information
  - Score = minimum I(A;B | S) over all conditioning sets S
  - If A and B are independent when you condition on the right stuff, the edge is fake

**What I got (Real Results from Day 9d):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| PMI | 0.398 | 1.000 | 0.557 |
| Day 9 screener | 0.639 | 0.962 | 0.761 |
| Pure CMI (k=1) | 0.917 | 1.000 | 0.954 |
| Pure CMI (k=2) | 0.945 | 1.000 | 0.970 |
| Pure CMI (k=3) | **0.966** | **1.000** | **0.981** |

**Single confound performance:**
- PMI: 36% precision (lots of false edges)
- Day 9 heuristic: 62% precision
- Pure CMI: **100% precision** ✓

**Double confound performance:**
- Pure CMI (k=2): 100% precision, 100% recall (perfect)
- Pure CMI (k=1): 83% precision (needs more context)

**Triple confound performance:**
- Pure CMI (k=3): 100% precision, 100% recall (still perfect)

**The honest part:**

This is **not** causal *direction* discovery. I'm not claiming "A causes B." It's more like "A and B probably have a direct edge, or they don't."

Also, it is concept-level, not fragment-scale. I'm working with conceptual variables that I defined, not discovering structure in learned latents.

## The Main Findings

| What We Tested | What Happened |
|---|---|
| Text + image in shared space | Worked (~99% recall) |
| Retrieval from memory | Stayed perfect even with noise |
| One vector for multiple concepts | Failed bad (1/6 retrieval) |
| Slot memory fix | Worked (6/6 in clean test) |
| Graph reasoning | Works when the seed is clean |
| Reasoning from noisy seed | Breaks around 40% noise |
| Refinement (multi-step) | Can correct wrong first guesses |
| Uncertainty (soft updates) | Can hold graded belief instead of collapsing |
| Context resolving ambiguity | Works via graph structure |
| Hebbian graph learning | Learns frequency, not truth |
| PMI graph learning | Better, but confounds fool it |
| Concept-level causal skeleton | Promising results on simple confounds |

## The Full Picture (Diagram)

```
            input (text / image)
                     |
                     v
         shared latent meaning-space
                     |
         +-----------+-----------+
         |           |           |
         v           v           v
    retrieval   slot memory   learned graph
         |           |           |
         +-----------+-----------+
                     |
                     v
         multi-step latent dynamics
         (propagation + refinement)
                     |
         +-----------+-----------+
         |                       |
         v                       v
    hard attractor        soft uncertainty
    (one winner)          (graded belief)
```

**How it compares to LSNN:**

```
LSNN:
  spike input → recurrent spiking neurons → adaptive state → spike output

WorldField:
  text/image → latent vectors → fragments/slots/graph → refined latent state
```

Both have the same vibe: sparse activity, recurrence, memory through dynamics. Different mechanisms, same spirit.

## What Needs to Happen Next (Spec)

To make this actually usable and runnable (the specification):

1. **Repo-level `requirements.txt`** – so people can install all dependencies
2. **Shared `worldfield/` package** – config, device selection, loading, metrics, graph utilities
3. **Automated smoke tests** – not just printed results, but actual test assertions
4. **Clean folder structure** – source code separate from generated artifacts
5. **Better naming** – be clear about what's proven and what's not (e.g., "concept-level causal skeleton recovery," not "causal learning")

Then, to make it showable:

6. **Live demo environment** – Jupyter notebook or CLI that runs small experiments in real-time
7. **Visual dashboard** – watch latent space, memory slots, graph activation evolve
8. **Real-world examples** – actual image-caption pairs, not just synthetic colored shapes
9. **Uncertainty visualization** – show soft belief evolution over refinement steps

## In Plain English

WorldField is my attempt to prove that different types of meaning can live in one shared internal space.

Here's the story: I start with text and images because that's the simplest multimodal case. I train both to land in the same latent space. Then I store observations as fragments, retrieve nearby ones, keep multiple concepts active simultaneously with slot memory, and learn which things relate through a co-activation graph.

**What worked:**

- Text and images genuinely align in one space
- Retrieval is rock solid
- Slot memory fixes the "one vector" problem
- Graphs can store relationships
- Reasoning can improve over multiple steps
- Soft updates let you keep uncertainty

**What didn't work:**

- One vector collapses under multiple concepts
- One-shot reasoning breaks with noise
- Hebbian learning only learns frequency
- Simple confound detection fails on complex cases

**The honest summary:**

I'm not claiming WorldField is AGI. I'm claiming it's a legit small prototype showing that multimodal perception, memory, reasoning, uncertainty, and early causal learning can all live inside a unified latent-space framework.

## What I Still Haven't Proven

- Real-world generalization (not just colored shapes)
- Audio or video
- Causal direction (only causal skeleton)
- Fragment-level causality (only concept-level)
- Full, calibrated uncertainty everywhere
- Open-ended text generation
- Lifelong learning with new concepts

That's all future work.

## TL;DR

Different modalities share one space → store fragments → retrieve sparse memories → slot memory for multi-concept state → graph edges for relations → multi-step refinement for reasoning → soft updates for uncertainty → conditional information for causal structure. Worked great on toy data. Next: make it real, automate tests, then show it off.

WorldField is like a map of meaning.

Text and images are roads into the map. Fragments are places on the map. Slots
keep several places active. The graph learns paths between places. Reasoning is
activation moving along those paths. Refinement is the map settling after a few
updates. Uncertainty is letting more than one path stay alive instead of forcing
one answer too early.

That is the whole idea.
