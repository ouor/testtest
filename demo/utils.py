from __future__ import annotations

import tempfile
import os
from urllib.parse import urljoin


def normalize_base_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        return "http://localhost:8000"
    if url.endswith("/"):
        url = url[:-1]
    return url


def join_api(base_url: str, path: str) -> str:
    base = normalize_base_url(base_url) + "/"
    return urljoin(base, path.lstrip("/"))


def write_temp_bytes(data: bytes, *, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path
