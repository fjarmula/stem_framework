import json
from pathlib import Path


def verify(submission_path: str, repo_root: str = ".", expected_path: str | None = None) -> dict:
    return _verify_matrix_submission(submission_path, expected_path)


def _verify_matrix_submission(submission_path: str, expected_path: str | None) -> dict:
    submission = json.loads(Path(submission_path).read_text(encoding="utf-8"))
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8")) if expected_path else {}

    answer_set = submission.get("answer_set", [])
    paths = submission.get("paths", [])
    if not isinstance(answer_set, list) or not all(isinstance(item, str) for item in answer_set):
        return _fail("Matrix final_artifact.answer_set must be a list of node id strings.", ["answer_set_mismatch", "incorrect_output"])
    if not isinstance(paths, list) or not all(isinstance(path, list) for path in paths):
        return _fail("Matrix final_artifact.paths must be a list of path lists.", ["path_trace_missing", "incorrect_output"])
    if not all(all(isinstance(node_id, str) for node_id in path) for path in paths):
        return _fail("Matrix final_artifact.paths must contain only node-id strings.", ["path_trace_missing", "incorrect_output"])

    terminal_nodes = sorted({path[-1] for path in paths if path})
    if terminal_nodes and sorted(answer_set) != terminal_nodes:
        return _fail(
            "Matrix answer_set must contain the terminal node ids from paths.",
            ["answer_set_mismatch", "path_trace_missing"],
        )
    if sorted(answer_set) != sorted(expected.get("answer_set", [])):
        return _fail(
            "The answer set does not match the task-owned graph verifier result.",
            ["answer_set_mismatch", "graph_traversal_failure"],
        )
    if sorted(paths) != sorted(expected.get("paths", [])):
        return _fail(
            "The path traces do not match the task-owned graph verifier result.",
            ["path_trace_missing", "graph_traversal_failure"],
        )
    return {"success": True, "critique": "The output contains the verified answer set and path traces.", "identified_gaps": []}


def _fail(critique: str, tags: list[str]) -> dict:
    return {"success": False, "critique": critique, "identified_gaps": tags}
