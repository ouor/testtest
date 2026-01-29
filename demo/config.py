from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoConfig:
    api_base_url: str
    timeout_seconds: int

    @staticmethod
    def from_env() -> "DemoConfig":
        api_base_url = (os.getenv("DEMO_API_BASE_URL") or "http://localhost:8000").strip()
        if api_base_url.endswith("/"):
            api_base_url = api_base_url[:-1]

        timeout_raw = (os.getenv("DEMO_TIMEOUT_SECONDS") or "60").strip()
        try:
            timeout_seconds = int(timeout_raw)
        except Exception:
            timeout_seconds = 60

        return DemoConfig(api_base_url=api_base_url, timeout_seconds=timeout_seconds)
