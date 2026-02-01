from __future__ import annotations

import os
from dataclasses import dataclass

from app.core.errors.exceptions import AppError


SYSTEM_PROMPT = """Task: Generate a concise, one-sentence descriptive caption for the provided movie information.
Instructions: Use the following structure:
[Genre] movie poster, [Main Subject & Action], [Mood/Expression], [Visual Style/Composition].

Constraints: Keep it short and punchy. Avoid flowery prose or unnecessary filler words (like 'This is a picture of...').
Focus on descriptive keywords that capture the essence of the image.
""".strip()


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    base_url: str | None

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise AppError(
                code="OPENAI_NOT_CONFIGURED",
                message="OPENAI_API_KEY is required to generate prompts",
                http_status=500,
            )

        model = (os.getenv("OPENAI_PROMPT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1").strip()
        if not model:
            model = "gpt-4.1"

        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
        return cls(api_key=api_key, model=model, base_url=base_url)


def generate_image_prompt(*, title: str, description: str) -> str:
    """Generate a single-sentence poster prompt using OpenAI GPT.

    This is a sync function (run it in a worker thread).
    """

    settings = OpenAISettings.from_env()

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise AppError(
            code="OPENAI_DEPENDENCY_MISSING",
            message="openai python package is required (pip install openai)",
            http_status=500,
            detail={"error": repr(exc)},
        ) from exc

    client_kwargs: dict[str, object] = {"api_key": settings.api_key}
    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url

    client = OpenAI(**client_kwargs)

    user_content = f"Title: {title.strip()}\nDescription: {description.strip()}"

    try:
        request_kwargs: dict[str, object] = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.4,
        }

        resp = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        raise AppError(
            code="OPENAI_REQUEST_FAILED",
            message="Failed to generate image prompt",
            http_status=502,
            detail={"error": repr(exc)},
        ) from exc

    try:
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        content = ""

    if not content:
        raise AppError(
            code="OPENAI_EMPTY_RESPONSE",
            message="OpenAI returned an empty prompt",
            http_status=502,
        )

    # Keep it one line.
    content = " ".join(content.split())

    # Remove wrapping quotes if the model returns them.
    if len(content) >= 2 and content[0] == '"' and content[-1] == '"':
        content = content[1:-1].strip()

    return content
