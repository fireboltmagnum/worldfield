"""WorldField — a cognitive architecture on a single shared latent space."""
from ._version import __version__
from .config import Config
from .device import pick_device
from .core import Engine, WorldGraph, SlotMemory, PMIGraph
from .core import ActivationEngine, WorldState, WorldStateBuilder
from .core import ContextManager, GoalManager
from .core import ContextWindow, ContextEvent, ConceptAttention, AttentionResult, MemoryRetrieval
from .reasoning import ReasoningEngine, format_answer, InferenceEngine
from .planning import Planner, PlanStep
from .simulation import Simulator
from .learning import LearningEngine

__all__ = [
    "__version__", "Config", "pick_device",
    "Engine", "WorldGraph", "SlotMemory", "PMIGraph",
    "ActivationEngine", "WorldState", "WorldStateBuilder",
    "ContextManager", "GoalManager",
    "ContextWindow", "ContextEvent", "ConceptAttention", "AttentionResult", "MemoryRetrieval",
    "ReasoningEngine", "format_answer", "InferenceEngine",
    "Planner", "PlanStep", "Simulator",
]
