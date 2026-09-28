"""Extract executable GUI action semantics from CoMEM's verbose responses."""

from __future__ import annotations

import json


OPERATIONAL_ARGUMENTS = frozenset({
    "element_id", "element_ids", "coords", "direction", "text", "answer",
    "query", "page_name", "button", "url",
})


def executable_action(response: str) -> str | None:
    """Return stable command JSON, discarding free-form chain-of-thought.

    A response may have prose before/after a fenced JSON object. Only a
    `name` plus `arguments` dictionary is accepted. Unknown argument fields
    are excluded rather than passing model-written reasoning into memory.
    """
    decoder = json.JSONDecoder()
    for index, character in enumerate(response):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        name, arguments = candidate.get("name"), candidate.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            continue
        command = {"name": name.strip().lower(), "arguments": {
            key: arguments[key] for key in sorted(arguments) if key in OPERATIONAL_ARGUMENTS
        }}
        return json.dumps(command, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return None
