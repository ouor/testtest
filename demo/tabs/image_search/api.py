from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from demo.http_client import HttpClient, HttpResult
from demo.utils import join_api


def upload_image(*, client: HttpClient, base_url: str, image_path: str) -> HttpResult:
    url = join_api(base_url, "/v1/images")
    path = Path(image_path)
    guessed, _ = mimetypes.guess_type(str(path))
    content_type = guessed or "image/jpeg"
    filename = path.name or "image.jpg"
    with open(image_path, "rb") as f:
        files = {"file": (filename, f, content_type)}
        return client.post_multipart(url, files=files, data={})


def list_images(*, client: HttpClient, base_url: str) -> HttpResult:
    url = join_api(base_url, "/v1/images")
    return client.get(url)


def search_images(*, client: HttpClient, base_url: str, payload: dict[str, Any]) -> HttpResult:
    url = join_api(base_url, "/v1/images/search")
    return client.post_json(url, payload)


def get_image_file_redirect(*, client: HttpClient, base_url: str, image_id: str, follow: bool) -> HttpResult:
    url = join_api(base_url, f"/v1/images/{image_id}/file")
    return client.get(url, allow_redirects=bool(follow))


def delete_image(*, client: HttpClient, base_url: str, image_id: str) -> HttpResult:
    url = join_api(base_url, f"/v1/images/{image_id}")
    return client.delete(url)
