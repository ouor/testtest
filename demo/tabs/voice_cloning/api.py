from __future__ import annotations

from typing import Any

from demo.http_client import HttpClient, HttpResult
from demo.utils import join_api


def post_generate_voice_mp3_multipart(
    *, client: HttpClient, base_url: str, ref_audio_path: str, fields: dict[str, Any]
) -> HttpResult:
    url = join_api(base_url, "/v1/voice/generate")
    with open(ref_audio_path, "rb") as f:
        files = {"ref_audio": ("ref.mp3", f, "audio/mpeg")}
        data = {k: str(v) for k, v in fields.items()}
        return client.post_multipart(url, files=files, data=data)


def post_generate_voice_to_r2(*, client: HttpClient, base_url: str, payload: dict[str, Any]) -> HttpResult:
    url = join_api(base_url, "/v1/r2/voice/generate")
    return client.post_json(url, payload)
