from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    headers: dict[str, str]
    body_bytes: bytes

    def json(self) -> Any:
        return json.loads(self.body_bytes.decode("utf-8"))


class HttpClient:
    """Small wrapper around requests with consistent errors.

    This module is standalone and does not import the server code.
    """

    def __init__(self, *, timeout_seconds: float):
        self._timeout_seconds = float(timeout_seconds)

    def get(self, url: str, *, allow_redirects: bool = True) -> HttpResult:
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("requests is required to run the demo. Install demo/requirements.txt") from exc

        resp = requests.get(url, timeout=self._timeout_seconds, allow_redirects=allow_redirects)
        return HttpResult(
            status_code=int(resp.status_code),
            headers={k.lower(): v for k, v in resp.headers.items()},
            body_bytes=resp.content,
        )

    def post_json(self, url: str, payload: dict[str, Any]) -> HttpResult:
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("requests is required to run the demo. Install demo/requirements.txt") from exc

        resp = requests.post(url, json=payload, timeout=self._timeout_seconds)
        return HttpResult(
            status_code=int(resp.status_code),
            headers={k.lower(): v for k, v in resp.headers.items()},
            body_bytes=resp.content,
        )

    def post_multipart(self, url: str, *, files: dict[str, Any], data: dict[str, Any]) -> HttpResult:
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("requests is required to run the demo. Install demo/requirements.txt") from exc

        resp = requests.post(url, files=files, data=data, timeout=self._timeout_seconds)
        return HttpResult(
            status_code=int(resp.status_code),
            headers={k.lower(): v for k, v in resp.headers.items()},
            body_bytes=resp.content,
        )

    def delete(self, url: str) -> HttpResult:
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("requests is required to run the demo. Install demo/requirements.txt") from exc

        resp = requests.delete(url, timeout=self._timeout_seconds)
        return HttpResult(
            status_code=int(resp.status_code),
            headers={k.lower(): v for k, v in resp.headers.items()},
            body_bytes=resp.content,
        )
