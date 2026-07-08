import json
from pathlib import Path
from typing import List, Dict, Any

from src.services.task_repository import load_task_source
from src.utils.config import config


class TaskLoader:
    def __init__(self, file_path: str = config["experiments"]["dir"]):
        self.file_path = Path(file_path)
        self.data: Dict[str, Any] = load_task_source(self.file_path)
        self.episodes: List[Dict[str, Any]] = self.data.get("episodes", [])

    @property
    def evolution_tasks(self) -> List[str]:
        return self._episode_prompts(split="train")

    @property
    def validation_tasks(self) -> List[str]:
        return self._episode_prompts(split="validation")

    @property
    def benchmark_name(self) -> str:
        return self.data.get("name", self.file_path.name)

    def _episode_prompts(self, split: str) -> List[str]:
        prompts: List[str] = []

        for episode in self.episodes:
            if episode.get("split") != split:
                continue
            prompts.append(self._render_episode_prompt(episode))
        return prompts

    @staticmethod
    def _render_episode_prompt(episode: Dict[str, Any]) -> str:
        episode_contract = episode.get("episode_contract", {})
        public_payload = {
            "benchmark_version": "2.0",
            "domain_id": episode.get("domain_id"),
            "domain_description": episode.get("domain_description"),
            "episode_id": episode.get("episode_id"),
            "split": episode.get("split"),
            "initial_prompt": episode.get("initial_prompt"),
            "required_capabilities": episode.get("required_capabilities", []),
            "allowed_failure_tags": episode.get("allowed_failure_tags", []),
            "episode_contract": {
                "episode_is_stateful": episode_contract.get("episode_is_stateful", True),
                "minimum_turns": episode_contract.get("minimum_turns", 5),
                "success_requires": episode_contract.get("success_requires", []),
                "private_verifier_artifacts": "withheld from agent",
            },
            "public_artifacts": episode.get("public_artifacts", {}),
            "artifact_manifest": episode.get("artifact_manifest", {}),
            "output_contract": episode.get("output_contract", {}),
            "clinical_probes": episode.get("clinical_probes", []),
            "turns": episode.get("turns", []),
            "baseline_expectation": episode.get("baseline_expectation", {}),
        }

        return (
            "STATEFUL STEM-CELL BENCHMARK EPISODE\n"
            "You must solve this episode using only public artifacts listed below. "
            "Private verifier artifacts are intentionally withheld.\n\n"
            f"{json.dumps(public_payload, indent=2)}\n\n"
            "Required final response format:\n"
            "- final_artifact: the completed answer, ledger, proof object, or answer set\n"
            "- state_trace: reconstructable intermediate state across all required turns\n"
            "- evidence: public artifact paths and/or execution traces used\n"
            "- limitations: explicit statement if the public task is impossible under its rules\n"
        )
