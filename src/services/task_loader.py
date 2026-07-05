import json
from pathlib import Path
from typing import List, Dict, Any
import yaml
from src.utils.config import config


class TaskLoader:
    def __init__(self, file_path: str = config["experiments"]["dir"]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Task file not found: {file_path}")
        with self.file_path.open("r") as f:
            self.data: Dict[str, Any] = yaml.safe_load(f)
        self.is_stateful_benchmark = "domains" in self.data and "benchmark_version" in self.data

    @property
    def evolution_tasks(self) -> List[str]:
        if self.is_stateful_benchmark:
            return self._episode_prompts(split="train")
        return self.data.get("evolution_set", [])

    @property
    def validation_tasks(self) -> List[str]:
        if self.is_stateful_benchmark:
            return self._episode_prompts(split="validation")
        return self.data.get("validation_set", [])

    @property
    def benchmark_name(self) -> str:
        if self.is_stateful_benchmark:
            return self.data.get("name", self.file_path.name)
        return self.file_path.name

    def _episode_prompts(self, split: str) -> List[str]:
        prompts: List[str] = []
        episode_contract = self.data.get("episode_contract", {})
        baseline_expectation = self.data.get("baseline_expectation", {})

        for domain in self.data.get("domains", []):
            for episode in domain.get("episodes", []):
                if episode.get("split") != split:
                    continue
                prompts.append(self._render_episode_prompt(
                    domain=domain,
                    episode=episode,
                    episode_contract=episode_contract,
                    baseline_expectation=baseline_expectation,
                ))
        return prompts

    @staticmethod
    def _render_episode_prompt(
            domain: Dict[str, Any],
            episode: Dict[str, Any],
            episode_contract: Dict[str, Any],
            baseline_expectation: Dict[str, Any],
    ) -> str:
        public_payload = {
            "benchmark_version": "2.0",
            "domain_id": domain.get("domain_id"),
            "domain_description": domain.get("description"),
            "episode_id": episode.get("episode_id"),
            "split": episode.get("split"),
            "initial_prompt": episode.get("initial_prompt"),
            "required_capabilities": domain.get("required_capabilities", []),
            "allowed_failure_tags": domain.get("failure_tags", []),
            "episode_contract": {
                "episode_is_stateful": episode_contract.get("episode_is_stateful", True),
                "minimum_turns": episode_contract.get("minimum_turns", 5),
                "success_requires": episode_contract.get("success_requires", []),
                "private_verifier_artifacts": "withheld from agent",
            },
            "public_artifacts": episode.get("public_artifacts", {}),
            "artifact_manifest": episode.get("artifact_manifest", {}),
            "turns": episode.get("turns", []),
            "baseline_expectation": baseline_expectation,
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
