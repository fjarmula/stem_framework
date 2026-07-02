import importlib.util
import json
from pathlib import Path


def load_parser(repo_root: Path):
    parser_path = repo_root / "benchmarks/security/sec_002/app/parser.py"
    spec = importlib.util.spec_from_file_location("sec_002_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(submission_path: str, repo_root: str = ".") -> bool:
    root = Path(repo_root)
    submission = json.loads(Path(submission_path).read_text())
    packet = submission.get("packet")
    observed = submission.get("observed_result")
    parser = load_parser(root)
    return parser.authorize(packet) == observed == "GRANTED:matrix-control"
