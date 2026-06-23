# Day One — Worldfield core-claim test

This is the smallest possible experiment that tests the central claim of the
[implementation plan](../worldfield_implementation_plan.md):

> Can text and image of the same thing be mapped into the **same** latent
> neighborhood?

Everything the plan defers is **deliberately absent** here: no fragment store,
no ANN retrieval, no refinement loop, no continual learning. Just two thin
encoders projecting into one shared space, trained with a contrastive alignment
loss and a reconstruction loss.

If matching pairs do **not** cluster in this minimal setup, none of the harder
machinery in the plan will save it. If they do, the core idea has legs and you
build outward from here (Stage A → B → ...).

## The dataset

Synthetic, generated on the fly — no downloads. Each sample is a colored shape
on a canvas plus a text description, e.g. a red square, a blue circle, a green
triangle. We *know* exactly what should align, so the experiment is honest:
"red square" image must land near the text "a red square".

This controls the variables. Real image-caption data comes later (plan §12).

## What it does

1. Generate paired (image, text) samples for combinations of {color} x {shape}.
2. Encode image (tiny CNN) and text (char-level GRU) into a shared `d`-dim space.
3. Train with:
   - **contrastive (InfoNCE)** — matching image/text pairs pulled together,
     mismatched pushed apart. This is the alignment claim.
   - **reconstruction** — image decoded back from its latent (proves the latent
     actually carries content, not just a matching trick).
4. Report hard metrics: cross-modal retrieval accuracy (R@1), and save a
   2D UMAP/PCA plot of the shared space colored by class.

## Run

```bash
cd day_one
./setup.sh          # one-time: creates .venv (Python 3.12) + installs deps
source .venv/bin/activate
python train.py     # trains, prints metrics, writes plots to ./out
```

Runs on CPU or Apple MPS in a couple of minutes. No GPU required.

## Known limitations (as of Day 4.5 — measured, not guessed)

The memory substrate is characterized, not perfect. Carry these forward:

- **effective capacity < slot count** — correlated concepts merge, so N slots
  hold fewer than N distinct concepts in practice.
- **routing threshold is sensitive** — too low merges distinct concepts, too
  high fails to merge genuine repeats; must be tuned to real similarity.
- **correlated concepts merge** under the default routing threshold.

Consequence for later experiments: run reasoning tests strictly WITHIN capacity
(e.g. 8 slots / 6 concepts, never 8 slots / 40), so a memory limit is never
mistaken for a reasoning failure.

## How to read the result

- **R@1 (image→text) and R@1 (text→image)** climbing well above chance
  (chance = 1/num_classes) = alignment is forming. This is the pass/fail signal.
- **out/latent_space.png** — if the shared space is real, points of the same
  class (regardless of modality) cluster together, and the matching image-dot
  and text-dot for a class sit near each other.
- **out/recon.png** — reconstructions should resemble inputs. Lossy is fine at
  `d=128`; we only need to confirm the latent carries content.

If R@1 stays near chance: stop and fix the core before adding anything.
