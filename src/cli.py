import argparse
from typing import Optional

from src.evaluation.stateful_benchmark import parse_episode_prompt
from src.utils.config import config


DOMAIN_ALIASES = {
    "all": None,
    "trading": "trading_floor",
    "trade": "trading_floor",
    "trading_floor": "trading_floor",
    "security": "security_sandbox",
    "security_sandbox": "security_sandbox",
    "matrix": "matrix_database",
    "matrix_database": "matrix_database",
}


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stem Cell evolutionary training.")
    parser.add_argument(
        "--domain",
        choices=sorted(DOMAIN_ALIASES),
        default="all",
        help=(
            "Restrict training/evaluation to one benchmark domain. "
            "Example: --domain trading trains on trade_001 and validates on trade_002."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=config["evolution"]["max_generations"],
        help="Maximum evolution epochs for this run.",
    )
    return parser.parse_args()


def parse_inference_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved Stem Cell genome on tasks or benchmark episodes.")
    parser.add_argument("--genome", default="mature_cell.json", help="Path to a saved genome JSON file.")
    parser.add_argument("--task", help="Single task prompt to run.")
    parser.add_argument("--task-file", help="Path to a text file containing a single task prompt.")
    parser.add_argument(
        "--benchmark",
        choices=["evolution", "validation", "all"],
        help="Run tasks from the configured task source instead of a single task.",
    )
    parser.add_argument("--no-verify", action="store_true", help="Do not run deterministic benchmark verification.")
    return parser.parse_args()


def task_label(task: str) -> str:
    """Return a compact label for large benchmark episode prompts."""
    payload = parse_episode_prompt(task)
    if payload:
        return str(payload.get("episode_id", "unknown_episode"))
    return task[:80]


def filter_tasks_by_domain(tasks: list[str], domain: Optional[str]) -> list[str]:
    if domain is None:
        return tasks
    filtered = []
    for task in tasks:
        payload = parse_episode_prompt(task)
        if payload and payload.get("domain_id") == domain:
            filtered.append(task)
    return filtered
