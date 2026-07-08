import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import config


def load_task_source(file_path: str | Path = config["experiments"]["dir"]) -> Dict[str, Any]:
    """Load benchmark task definitions from the task manifest directory."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Task directory not found: {path}")
    if not path.is_dir():
        raise ValueError(
            f"Task source must be a directory of per-episode manifests, not a file: {path}"
        )
    return _load_task_directory(path)


def find_task_episode(
    domain_id: str,
    episode_id: str,
    file_path: str | Path = config["experiments"]["dir"],
) -> Optional[Dict[str, Any]]:
    for episode in load_task_source(file_path).get("episodes", []):
        if episode.get("domain_id") == domain_id and episode.get("episode_id") == episode_id:
            return episode
    return None


def _load_task_directory(path: Path) -> Dict[str, Any]:
    manifest_path = path / "benchmark.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {}

    episodes = []
    for task_path in sorted(path.glob("*/*.json")):
        with task_path.open("r", encoding="utf-8") as handle:
            episode = json.load(handle)
        if task_path.name == "benchmark.json":
            continue
        if episode.get("benchmark_version") != "2.0":
            raise ValueError(f"Task file {task_path} must declare benchmark_version=2.0")
        episodes.append(episode)

    if not episodes:
        raise ValueError(f"Task directory {path} contains no v2 task episode files")

    return {
        "benchmark_version": manifest.get("benchmark_version", "2.0"),
        "name": manifest.get("name", path.name),
        "baseline_expectation": manifest.get("baseline_expectation", {}),
        "episode_contract": manifest.get("episode_contract", {}),
        "episodes": episodes,
    }
