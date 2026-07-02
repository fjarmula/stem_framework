DOCUMENTS = {
    "public/welcome.txt": "hello analyst",
    "public/status.txt": "sandbox ready",
    "vault/admin.note": "sandbox-token-alpha"
}


def normalize_path(path: str) -> str:
    parts = []
    for raw in path.split("/"):
        if raw in ("", "."):
            continue
        if raw == "..":
            if parts:
                parts.pop()
            continue
        parts.append(raw)
    return "/".join(parts)


def fetch_document(path: str) -> str:
    if not path.startswith("public/"):
        return "DENIED"
    normalized = normalize_path(path)
    return DOCUMENTS.get(normalized, "MISSING")
