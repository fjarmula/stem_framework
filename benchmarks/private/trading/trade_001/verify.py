import json
from pathlib import Path


def verify(submission_path: str, repo_root: str = ".", expected_path: str | None = None) -> dict:
    return _verify_trading_submission(submission_path, expected_path)


def _verify_trading_submission(submission_path: str, expected_path: str | None) -> dict:
    submission = json.loads(Path(submission_path).read_text(encoding="utf-8"))
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8")) if expected_path else {}

    portfolio = submission.get("final_portfolio", {})
    ledger = submission.get("ledger", [])
    if not isinstance(portfolio, dict) or "positions" not in portfolio:
        return _fail("Trading final_portfolio must contain cash and nested positions.", ["incomplete_final_artifact"])
    if not isinstance(ledger, list):
        return _fail("Trading ledger must be a list of transaction row objects.", ["incomplete_final_artifact"])

    required_ledger_keys = {"tick", "asset", "side", "quantity", "price", "fee", "cash_after"}
    for index, row in enumerate(ledger):
        if not isinstance(row, dict):
            return _fail(f"Trading ledger row {index} must be an object.", ["ledger_mismatch", "state_tracking_failure"])
        missing = sorted(required_ledger_keys - set(row))
        if missing:
            return _fail(
                f"Trading ledger row {index} is missing required keys {missing}. "
                f"Every row must include {sorted(required_ledger_keys)}.",
                ["ledger_mismatch", "state_tracking_failure"],
            )

    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash")
    if positions != expected.get("final_positions") or cash != expected.get("final_cash"):
        return _fail(
            "The final portfolio does not match the task-owned ledger verifier result.",
            ["ledger_mismatch", "incorrect_output"],
        )
    if _normalize_ledger(ledger) != _normalize_ledger(expected.get("ledger", [])):
        return _fail(
            "The transaction ledger rows do not match the task-owned verifier trace.",
            ["ledger_mismatch", "state_tracking_failure"],
        )
    note = expected.get("note")
    if note and "impossible" not in str(submission.get("limitations", "")).lower():
        return _fail("The output reaches the best legal ledger but does not emit the required impossibility limitation.", ["incomplete_final_artifact"])
    return {"success": True, "critique": "The output contains the verified final portfolio, ledger, and state trace.", "identified_gaps": []}


def _normalize_ledger(ledger: object) -> object:
    if not isinstance(ledger, list):
        return ledger
    numeric_keys = {"tick", "quantity", "price", "fee", "cash_after"}
    normalized = []
    for row in ledger:
        if not isinstance(row, dict):
            normalized.append(row)
            continue
        normalized.append({
            key: _normalize_number(value) if key in numeric_keys else value
            for key, value in row.items()
        })
    return normalized


def _normalize_number(value: object) -> object:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _fail(critique: str, tags: list[str]) -> dict:
    return {"success": False, "critique": critique, "identified_gaps": tags}
