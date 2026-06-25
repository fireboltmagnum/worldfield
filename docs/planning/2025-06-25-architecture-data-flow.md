# WorldField Architecture — Data Flow

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│  Text  │  Image  │  Audio  │  Video  │  Sensor  │  ...          │
└────────┼─────────┼─────────┼─────────┼──────────┼──────────────┘
         │         │         │         │          │
         ▼         ▼         ▼         ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      UNDERSTANDING                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Parser  │  │  Image   │  │  Audio   │  │  Video   │        │
│  │  (spaCy) │  │  Encoder │  │  Encoder │  │  Encoder │  ...    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       └─────────────┼─────────────┼─────────────┘               │
│                     │             │                              │
│                     ▼             ▼                              │
│              ┌────────────────────────┐                          │
│              │  Concept Extraction     │                         │
│              │  (modality-agnostic)    │                         │
│              └───────────┬────────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ACTIVATION LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ActivationManager                                        │   │
│  │  • seed(concepts) → activation=1.0                       │   │
│  │  • spread(hops, decay) → traverse graph                  │   │
│  │  • decay(time) → reduce activations                      │   │
│  │  • threshold(min) → prune weak                           │   │
│  │  • get_active() → {concept: activation}                  │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│                  Active Concepts                                  │
│         cat=0.97  sofa=0.91  sleep=0.88  animal=0.82            │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WORLD STATE                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  WorldStateBuilder                                        │   │
│  │  • entities: {cat: 0.97, sofa: 0.91, ...}               │   │
│  │  • relations: [cat ─[sleeping_on]→ sofa (0.92)]          │   │
│  │  • attributes: {cat: {black: 0.75}}                      │   │
│  │  • alternatives: [{cat ─[sitting_on]→ sofa (0.44)}]     │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│              Current Reality Model                               │
│         (modality-independent, hypothesis-based)                 │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTEXT LAYER (Phase 5)                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ContextManager                                           │   │
│  │  • topic: current discussion topic                       │   │
│  │  • recent_concepts: sliding window                       │   │
│  │  • working_set: persistently activated concepts          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│                      GOAL LAYER (Phase 6)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  GoalManager                                              │   │
│  │  • goals: prioritized goal stack                         │   │
│  │  • current_task: active subtask                          │   │
│  │  • blockers: dependencies blocking progress              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│                      PLANNING (Phase 7)                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Planner                                                  │   │
│  │  • plan(goal, state) → list[Step]                        │   │
│  │  • replan(failed_step) → list[Step]                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│                      REASONING + SIMULATION (Phase 7)              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  InferenceEngine + Simulator                              │   │
│  │  • inherit_properties(world_state, graph)                 │   │
│  │  • compose_relations(world_state, graph)                  │   │
│  │  • detect_contradictions(world_state)                     │   │
│  │  • propagate_confidence(inferences)                       │   │
│  │  • build_explanation(chain)                               │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────┐                  │   │
│  │  │  ExplanationChain                    │                  │   │
│  │  │  cat ─[is_a]→ mammal ─[is_a]→ animal│                  │   │
│  │  │  Conclusion: cat is an animal       │                  │   │
│  │  │  Confidence: 0.91                   │                  │   │
│  │  └─────────────────────────────────────┘                  │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│              Updated World State (with inferences)               │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LANGUAGE GENERATION                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  State   │───▶│    Prompt    │───▶│     Decoder          │   │
│  │Serializer│    │  Constructor │    │  (T5-small / API)    │   │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘   │
│                                                 │                │
│                                                 ▼                │
│                                         Generated Text          │
│                                    "A black cat is sleeping     │
│                                     on the sofa."               │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTINUOUS LEARNING                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LearningEngine                                          │   │
│  │  • update_confidence(observation) → reinforce/weaken     │   │
│  │  • resolve_contradictions(world_state) → merge/prune    │   │
│  │  • prune_memory(threshold) → remove low-confidence      │   │
│  │  • refine_concepts(new_evidence) → update aliases       │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│                    WORLD GRAPH                                    │
│           (persistent knowledge with confidence)                  │
└──────────────────────────────────────────────────────────────────┘

## Modality Independence

All modalities converge on the same WORLD STATE:

```
Text:  "The black cat is sleeping on the sofa."
Image: [pixel data]
       │
       ▼
       │
       ┌──────────────────────────────┐
       │  WORLD STATE (shared)        │
       │                              │
       │  Entities:                   │
       │    cat (0.97)                │
       │    sofa (0.91)               │
       │                              │
       │  Relations:                  │
       │    cat ─[sleeping_on]→ sofa  │
       │                              │
       │  Attributes:                 │
       │    cat → black               │
       └──────────────────────────────┘
```

## Dependency Graph

```
activation.py ──── depends on ────> world_graph.py
world_state.py ── depends on ────> activation.py, world_graph.py
inference.py ──── depends on ────> world_state.py, world_graph.py
explanation.py ── depends on ────> inference.py
serializer.py ─── depends on ────> world_state.py
decoder.py ────── depends on ────> serializer.py
learning.py ───── depends on ────> world_graph.py, world_state.py
```

Build order: activation → world_state → inference → explanation → serializer → decoder → learning
