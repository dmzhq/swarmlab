"""swarmlab — deterministic, replayable multi-agent runs across any LLM provider."""

from __future__ import annotations

from swarmlab.dag import AgentSpec, DAGSpec, EdgeSpec, ToolSpec
from swarmlab.loaders import DAGValidationError, load_python, load_yaml
from swarmlab.providers import CompletionResult, Message, Provider, get_provider
from swarmlab.seed import compute_invocation_seed
from swarmlab.store import ContentAddressedStore, Entry

__version__ = "0.0.1"
__all__ = [
    "AgentSpec",
    "CompletionResult",
    "ContentAddressedStore",
    "DAGSpec",
    "DAGValidationError",
    "EdgeSpec",
    "Entry",
    "Message",
    "Provider",
    "ToolSpec",
    "__version__",
    "compute_invocation_seed",
    "get_provider",
    "load_python",
    "load_yaml",
]
