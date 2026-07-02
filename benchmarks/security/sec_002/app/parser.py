def parse_packet(packet: str) -> dict:
    fields = {}
    for chunk in packet.split(";"):
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def authorize(packet: str) -> str:
    fields = parse_packet(packet)
    if fields.get("sig") != "sandbox":
        return "DENIED:bad-signature"
    if fields.get("cmd") != "view":
        return "DENIED:bad-command"
    if fields.get("role") == "admin":
        return "GRANTED:matrix-control"
    return "DENIED:wrong-role"
