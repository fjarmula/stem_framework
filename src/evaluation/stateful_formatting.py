import json

from src.evaluation.stateful_contract import extract_output_object


def format_stateful_output(agent_output: str) -> str:
    """Return a readable console view of a benchmark answer."""
    output = extract_output_object(agent_output)
    if output is None:
        return agent_output

    compact = {
        "final_artifact": output.get("final_artifact", output),
        "state_trace": output.get("state_trace", []),
        "evidence": output.get("evidence", []),
        "limitations": output.get("limitations", "none"),
    }
    return json.dumps(compact, indent=2, sort_keys=True)
