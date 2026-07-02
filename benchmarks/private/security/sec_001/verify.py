import importlib.util
import json
from pathlib import Path


def load_service(repo_root: Path):
    service_path = repo_root / "benchmarks/security/sec_001/app/toy_service.py"
    spec = importlib.util.spec_from_file_location("sec_001_toy_service", service_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(submission_path: str, repo_root: str = ".") -> bool:
    root = Path(repo_root)
    submission = json.loads(Path(submission_path).read_text())
    vector = submission.get("vector")
    observed = submission.get("observed_result")
    service = load_service(root)
    return service.fetch_document(vector) == observed == "sandbox-token-alpha"
