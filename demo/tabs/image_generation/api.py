from __future__ import annotations

from typing import Any

from demo.http_client import HttpClient, HttpResult
from demo.utils import join_api


def post_generate_image_png(*, client: HttpClient, base_url: str, payload: dict[str, Any]) -> HttpResult:
    url = join_api(base_url, "/v1/images/generate")
    return client.post_json(url, payload)


def post_generate_image_to_r2(*, client: HttpClient, base_url: str, payload: dict[str, Any]) -> HttpResult:
    url = join_api(base_url, "/v1/r2/images/generate")
    return client.post_json(url, payload)
