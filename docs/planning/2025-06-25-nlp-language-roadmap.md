# WorldField NLP + Language Generation Roadmap

**Date:** 2025-06-25
**Status:** Draft — awaiting review

## Vision

WorldField is not a fact engine or graph chatbot. It is a cognitive system that builds and updates an internal world model. The language system is an **interface** to that world model, not the intelligence itself.

The pipeline (phased):

```
Input
↓
Understanding
↓
Activation
↓
World State
↓
[Context Layer]    ← Phase 5
↓
[Goal Layer]       ← Phase 6
↓
[Planning]         ← Phase 6
↓
Reasoning
↓
[Simulation]       ← Phase 7
↓
Language Generation
↓
Learning
```

**Current priority (Phases 1-4):** Understanding → Activation → World State → Reasoning → Language → Learning
**Future (Phases 5-7):** Context → Goals → Planning → Simulation

## Architecture Principles

1. **The graph is memory.** It stores persistent knowledge with confidence and provenance.
2. **The reasoning engine is cognition.** It operates on the current world state, not directly on the graph.
3. **The language generator is communication.** It projects world state into text, not facts into templates.
4. **Every modality feeds the same world state.** Text, image, audio, video all update the same internal reality model.
5. **The system reasons with beliefs, not facts.** Uncertainty, competing hypotheses, and confidence scores are first-class.

---

## Phase 1: Improved NLP Pipeline

### Goal
Reliable concept extraction from text: objects, attributes, actions, relationships, temporal info, confidence.

### Current State
- spaCy-based parser handles basic SVO extraction
- Cross-sentence coreference resolution (rule-based)
- Negation detection
- Phrasal verb handling (prt + pobj)
- Copula / is-a extraction

### Needed Upgrades

| Component | Current | Target |
|-----------|---------|--------|
| Parser | SVO triples | Full thematic role labeling (agent, patient, instrument, location) |
| Entity resolution | Basic surface matching | Context-aware disambiguation |
| Temporal extraction | None | Tense, aspect, duration, ordering |
| Quantification | None | Countable, uncountable, quantifiers |
| Modality | Text only | Modality-agnostic concept representation |

### Deliverable
`worldfield/nlp/` — upgraded parser and extractor producing structured concept activations.

---

## Phase 2: Concept Activation Layer

### Goal
Runtime concept activations that spread through the graph and decay over time. This is the system's "working memory."

### Why Separate from the Graph
The graph is persistent storage. Activations are ephemeral, context-dependent, and represent what the system is currently "thinking about."

### Architecture

```
Input Concepts
↓
Seed Activations (directly extracted concepts → activation = 1.0)
↓
Spreading Activation (traverse graph edges, propagate with decay)
↓
Activation Decay (time-based, context-switch based)
↓
Activation Thresholding (prune below-threshold concepts)
```

### Components

**ActivationManager** (`worldfield/core/activation.py`)
- `seed(concepts: list[str])` — set initial activations from extraction
- `spread(hops: int = 2, decay: float = 0.5)` — traverse graph, propagate
- `decay(factor: float = 0.9)` — reduce all activations
- `threshold(min_val: float = 0.1)` — prune weak activations
- `get_active(threshold: float = 0.0) -> dict[str, float]` — current activations
- `reset()` — clear working memory

**Behavior:**
- Directly extracted concepts get activation = 1.0
- Each hop through the graph multiplies by decay factor
- Multiple paths to the same concept sum (with normalization)
- Context switch (new input) triggers decay on previous activations
- Below-threshold concepts are pruned but not removed from graph

### Data Flow
```
Input: "The black cat is sleeping on the sofa"
→ Extract: [cat(1.0), black(1.0), sleep(1.0), sofa(1.0)]
→ Spread 1 hop: animal(0.5), furniture(0.5), mat(0.3)
→ Spread 2 hops: pet(0.25), mammal(0.25), household_item(0.15)
→ Threshold (0.2): keep cat, black, sleep, sofa, animal, furniture, pet, mammal
→ Return activations for WORLD STATE
```

### Deliverable
`worldfield/core/activation.py` — ActivationManager class with full test suite.

---

## Phase 3: World State Builder

### Goal
Convert concept activations into a temporary reality model — the system's current "understanding" of the world.

### Why Separate from the Graph
The world state is:
- Ephemeral (resets or evolves with context)
- Hypothesis-based (multiple competing interpretations)
- Modality-independent (same structure for text, image, audio)

### Architecture

```
Activations
↓
Entity Tracker → [cat, sofa, ...]
↓
Relation Tracker → [cat ─[sleeping_on]→ sofa]
↓
Attribute Tracker → [cat → black]
↓
Confidence Estimator → each entity/relation gets a confidence
```

### Components

**WorldState** (`worldfield/core/world_state.py`)
- `entities: dict[str, float]` — tracked objects with confidence
- `relations: list[Relation]` — tracked relations with confidence
- `attributes: dict[str, dict[str, float]]` — entity → attribute → confidence
- `alternative_hypotheses: list[WorldState]` — competing interpretations

**WorldStateBuilder** (`worldfield/core/world_state.py`)
- `from_activations(activations: dict[str, float], relations: list[dict], graph) -> WorldState`
- `merge(existing: WorldState, new: WorldState) -> WorldState`
- `from_image(image_features) -> WorldState` (future)

### Relation Structure
```python
@dataclass
class Relation:
    source: str
    predicate: str
    target: str
    confidence: float
    source_modality: str  # "text", "image", etc.
    evidence_ids: list[str]  # observation IDs
```

### Data Flow
```
Activations: {cat: 0.97, sofa: 0.91, sleep: 0.88, ...}
+ Extracted relations: [cat ─[sleeping_on]→ sofa]
+ Graph knowledge: cat ─[is_a]→ animal (0.97)

→ World State:
  Entities: cat(0.97), sofa(0.91), sleep(0.88)
  Relations: cat ─[sleeping_on]→ sofa (0.92)
             cat ─[sitting_on]→ sofa (0.44) [alternative]
  Attributes: cat → black (0.75)
```

### Deliverable
`worldfield/core/world_state.py` — WorldState + WorldStateBuilder with full test suite.

---

## Phase 4: Reasoning Over World State

### Goal
Reasoning operates on the current world state, not on raw graph queries. This enables multi-step inference, contradiction detection, and explanation chains.

### Current Limitations
- Reasoner traverses graph directly (graph queries)
- No concept of "current situation"
- Inferences are graph lookups, not novel conclusions

### Architecture

```
World State
↓
Inference Engine
├── Property inheritance (cat is_a mammal → has fur)
├── Relation composition (A ─[located_on]→ B + B ─[part_of]→ C → A ─[located_on]→ C)
├── Contradiction detection (A ─[sleeping_on]→ B vs A ─[sitting_on]→ B)
├── Confidence propagation (combine confidences along inference chain)
└── Explanation chain construction
↓
Updated World State (with inferences)
```

### Components

**InferenceEngine** (`worldfield/reasoning/inference.py`)
- `inherit_properties(world_state, graph) -> list[Inference]`
- `compose_relations(world_state, graph) -> list[Inference]`
- `detect_contradictions(world_state) -> list[Contradiction]`
- `propagate_confidence(world_state, inferences) -> WorldState`
- `build_explanation(inference) -> ExplanationChain`

**ExplanationChain** (`worldfield/reasoning/explanation.py`)
```python
@dataclass
class ExplanationChain:
    premise: str
    steps: list[ExplanationStep]
    conclusion: str
    confidence: float
    
@dataclass
class ExplanationStep:
    rule: str
    source: str
    target: str
    confidence: float
```

### Deliverable
`worldfield/reasoning/inference.py`, `worldfield/reasoning/explanation.py` — inference engine + explanation chains.

---

## Phase 5: Language Generation

### Goal
Concepts should speak. The language generator projects the current world state (including reasoning results) into natural language.

### Not a Template System
The current `format_answer()` produces template-based responses. The goal is a learned decoder that takes concept activations + world state and produces fluent text.

### Options

| Option | Complexity | Quality | Dependencies |
|--------|-----------|---------|-------------|
| A: Tiny Transformer Decoder | High | Medium | PyTorch, custom training |
| B: T5-small fine-tune | Medium | High | transformers, concept→text pairs |
| C: Small instruction-tuned model | Low | High | transformers, API |

### Near-term Recommendation: Option C
Use an existing lightweight instruction-tuned model (e.g., FLAN-T5-small) as the decoder. Serialize world state + reasoning chain into a structured prompt. The model converts it to fluent text.

### Architecture

```
World State
↓
State Serializer
↓
[Structured text: entities, relations, attributes, inferences]
↓
Prompt Constructor
↓
[Prompt: "Given this world state:\n...\nGenerate a response."]
↓
Decoder (T5-small / etc.)
↓
[Generated text]
```

### Components

**StateSerializer** (`worldfield/nlg/serializer.py`)
- Serialize WorldState → structured text
- Serialize ExplanationChain → evidence text
- Serialize graph provenance → source text

**PromptConstructor** (`worldfield/nlg/prompts.py`)
- Build prompt from serialized state
- Inject evidence and confidence
- Control response style (concise, detailed, etc.)

**Decoder** (`worldfield/nlg/decoder.py`)
- Abstraction over model types
- `generate(state: WorldState, style: str) -> str`

### Deliverable
`worldfield/nlg/` — serializer, prompt constructor, decoder abstraction.

---

## Phase 5: Context Layer (Future — Do Not Build Yet)

### Problem
Without context, every input is processed independently. The system cannot maintain a topic, track discussion history, or know what's important right now.

### What It Stores
- **Current Topic** — what the conversation is about (e.g., "Language Generation")
- **Recent Concepts** — concepts mentioned recently (e.g., decoder, reasoning, graph)
- **Working Set** — concepts with elevated activation that persist across turns
- **Interaction History** — recent inputs and system responses

### Why It Matters
Without context:
```
User: "What about confidence propagation?"
System: starts fresh, doesn't know we were discussing reasoning
```

With context:
```
User: "What about confidence propagation?"
System: "Building on our discussion of the reasoning engine,
        confidence propagation works by..."
```

### Components
**ContextManager** (`worldfield/core/context.py`)
- `topic: str` — current discussion topic
- `recent_concepts: deque[str]` — sliding window of mentioned concepts
- `working_set: dict[str, float]` — persistently activated concepts
- `history: list[dict]` — recent interaction turns

### Deliverable
`worldfield/core/context.py` — ContextManager class.

---

## Phase 6: Goal Layer (Future — Do Not Build Yet)

### Problem
Without goals, the system reacts but does not act. It thinks, forgets, thinks, forgets — no direction.

### What It Stores
- **Primary Goal** — the main objective (e.g., "Build reasoning engine")
- **Subgoals** — decomposed steps (e.g., "Implement inheritance")
- **Current Task** — what's being worked on now
- **Blocked By** — dependencies that need resolving
- **Progress** — completed tasks, verification status

### Why It Matters
Goals turn the system from reactive to proactive. Instead of waiting for input, it can execute multi-step plans, verify its own work, and maintain direction across sessions.

### Components
**GoalManager** (`worldfield/core/goals.py`)
- `goals: list[Goal]` — prioritized goal stack
- `current_task: str` — active subtask
- `blockers: list[str]` — what's preventing progress
- `verify()` — check if current task is complete

### Deliverable
`worldfield/core/goals.py` — GoalManager class.

---

## Phase 7: Planning + Simulation (Future — Do Not Build Yet)

### Problem
Current reasoning is inferential (what IS true), not predictive (what WILL be true). Planning and simulation are what transform a knowledge system into an autonomous agent.

### Planning
The planner decomposes goals into executable steps:

```
Goal: Build decoder
Planner:
  1. Build serializer
  2. Build prompt constructor
  3. Build decoder wrapper
  4. Test
  5. Benchmark
```

**Planner** (`worldfield/planning/planner.py`)
- `plan(goal: Goal, state: WorldState) -> list[Step]`
- `replan(failed_step: Step) -> list[Step]`

### Simulation
The simulator predicts possible futures based on the current world state:

```
Current: cat sleeping on sofa
Simulation:
  - If cat jumps → sofa becomes occupied, cat becomes active
  - If dog enters → possible interaction (play/fight/flee)
  - If cat sleeps → location remains same, time passes
```

**Simulator** (`worldfield/simulation/engine.py`)
- `simulate(world_state, actions) -> list[WorldState]` — branch predictions
- `rank_outcomes(outcomes) -> list[WorldState]` — probability-weighted futures

### Why Not Now
Building simulation on top of a system that can't reliably reason yet would produce unreliable predictions. Simulation is the roof, not the foundation.

### Deliverable
`worldfield/planning/`, `worldfield/simulation/` — planner and simulator.

---

## Phase 8: Continuous Learning

### Goal
No epochs, no retraining loops. Learning updates the graph continuously as the system processes input.

### Principles
- Every input is a training example
- Confidence updates are immediate
- Contradictions trigger resolution
- Memory pruning prevents unbounded growth

### Components

**LearningEngine** (`worldfield/learning/engine.py`)
- `update_confidence(observation) -> None` — reinforce or weaken relations
- `resolve_contradictions(world_state) -> list[Resolution]` — handle conflicting evidence
- `prune_memory(threshold: float) -> int` — remove low-confidence/low-support edges
- `refine_concepts(new_evidence) -> None` — update concept aliases and vectors

### Deliverable
`worldfield/learning/` — continuous learning engine.

---

## Subsystem Breakdown

| Subsystem | Files | Dependencies | Priority |
|-----------|-------|-------------|----------|
| Activation | `core/activation.py` | Graph | P0 |
| World State | `core/world_state.py` | Activations, Graph | P0 |
| Inference | `reasoning/inference.py` | World State, Graph | P0 |
| Explanations | `reasoning/explanation.py` | Inferences | P1 |
| NLG Serializer | `nlg/serializer.py` | World State | P1 |
| NLG Decoder | `nlg/decoder.py` | Serializer | P1 |
| Learning | `learning/engine.py` | Graph, World State | P2 |

## Success Criteria

Near-term success is NOT "WorldField sounds like ChatGPT."

Near-term success IS:
- Understand text reliably (Phase 1)
- Build world states from concepts (Phase 2-3)
- Reason over world states (Phase 4)
- Explain reasoning (Phase 4)
- Generate natural language from concepts (Phase 5)
- Learn continuously (Phase 6)
- Stay fast (all phases)

The language system is an interface. The world model remains the intelligence.

## Benchmark Tasks

| Benchmark | Measures | Target |
|-----------|----------|--------|
| Concept extraction accuracy | Precision/recall on labeled corpus | >0.85 |
| Relation extraction accuracy | Precision/recall on labeled corpus | >0.80 |
| World state correctness | Human evaluation of state accuracy | >0.80 |
| Reasoning depth | Max inference chain length | >3 hops |
| Explanation quality | Human rating (1-5) | >3.5 |
| Language fluency | Human rating (1-5) | >3.0 |
| Learning speed | Time to update from new evidence | <100ms |
| Memory growth rate | Relations per day | <10% |
| Query latency | End-to-end response time | <500ms |
