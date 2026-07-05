"""
Compatibility facade for the stateful benchmark runtime.

The implementation is split across focused modules:
- stateful_contract.py: shared types, domain constants, prompt parsing, JSON helpers
- stateful_runner.py: physical multi-turn environment loop
- stateful_verifier.py: deterministic verifier and failure feedback
- stateful_formatting.py: console formatting helpers
"""

from src.evaluation.stateful_contract import (
    SUPPORTED_STATEFUL_DOMAINS,
    EpisodeRunResult,
    TurnExecutor,
    parse_episode_prompt,
)
from src.evaluation.stateful_formatting import format_stateful_output
from src.evaluation.stateful_runner import StatefulEpisodeRunner, run_stateful_episode
from src.evaluation.stateful_verifier import (
    unverifiable_inference_feedback,
    verify_physical_episode,
    verify_stateful_episode,
)

__all__ = [
    "SUPPORTED_STATEFUL_DOMAINS",
    "EpisodeRunResult",
    "TurnExecutor",
    "StatefulEpisodeRunner",
    "format_stateful_output",
    "parse_episode_prompt",
    "run_stateful_episode",
    "unverifiable_inference_feedback",
    "verify_physical_episode",
    "verify_stateful_episode",
]
