---
title: WorldField Demo
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# WorldField: Text + Image Alignment Demo

Interactive demo of **WorldField** — a research project testing whether text, images, memory, and reasoning can all share one unified latent space.

## What This Does

Upload an image or type text to see:
- How embeddings align in a shared space
- Which concepts match your input
- Similarity scores

## The Research

WorldField is a 9-day research journal exploring:
1. **Direct multimodality:** Can text + images share one latent space?
2. **Fragment retrieval:** Can we efficiently retrieve related memories?
3. **Slot memory:** Can we keep multiple concepts active simultaneously?
4. **Graph reasoning:** Can concepts learn relationships?
5. **Refinement:** Can systems correct wrong first guesses?
6. **Uncertainty:** Can systems hold graded belief instead of collapsing?
7. **Causality:** Can graphs distinguish confounds from real edges?

**Current Status:** Concept-level causal skeleton recovery (not fragment-scale, not causal direction)

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

## Links

- [Full Explanation](https://github.com/Ultimateclaudecoder/WorldField/blob/main/WORLD_FIELD_FULL_EXPLANATION.md)
- [GitHub Repo](https://github.com/Ultimateclaudecoder/WorldField)
- [Day-by-Day Research Journal](https://github.com/Ultimateclaudecoder/WorldField/tree/main/day_one)

---

**Note:** This is a demo version using mock embeddings. To use the real trained model, point `app.py` at the checkpoint file.
