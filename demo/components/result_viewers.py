from __future__ import annotations

import base64
from typing import Any


def bytes_to_data_url(mime_type: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def safe_json_or_text(data: bytes) -> Any:
    if not data:
        return {}
    try:
        import json

        return json.loads(data.decode("utf-8"))
    except Exception:
        try:
            text = data.decode("utf-8", errors="replace").strip()
            if not text:
                return {}
            return text
        except Exception:
            return {"error": "failed to decode response"}
