from __future__ import annotations

import io
import os
import uuid
from functools import partial

import anyio
import torch
from fastapi import Request

from app.core.errors.exceptions import InferenceError, ModelLoadError, OutOfMemoryError

from app.core.errors.exceptions import AppError
from app.core.llm.openai_prompter import generate_image_prompt
from app.domains.image_generation.schemas import GeneratePlayPosterRequest, GeneratePlayPosterToR2Request
from app.domains.image_generation.model import IMAGE_MODEL_KEY


FIXED_WIDTH = 768
FIXED_HEIGHT = 1152
FIXED_STEPS = 9
FIXED_SEED = 42
FIXED_GUIDANCE_SCALE = 0.0


def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip()
    if not prefix:
        return ""
    # Allow users to set either "AI/POSTER" or "AI/POSTER/".
    if not prefix.endswith("/"):
        prefix += "/"
    # Avoid accidental leading slash which would create odd-looking keys.
    prefix = prefix.lstrip("/")
    return prefix


def _image_remote_prefix() -> str:
    return _normalize_prefix(os.getenv("IMAGE_REMOTE_PREFIX", ""))


def _apply_prefix(*, prefix: str, key: str) -> str:
    key = (key or "").lstrip("/")
    if not prefix:
        return key
    if key.startswith(prefix):
        return key
    return f"{prefix}{key}"


async def generate_image_png(request: Request, payload: GeneratePlayPosterRequest) -> bytes:
    model = request.app.state.models.get(IMAGE_MODEL_KEY)
    if model is None:
        raise ModelLoadError("Image model is not loaded")

    try:
        limit = request.app.state.limits.get(IMAGE_MODEL_KEY)
    except Exception as exc:
        raise ModelLoadError("Concurrency limits not initialized") from exc

    # Generate the text prompt via OpenAI before acquiring the GPU semaphore.
    try:
        prompt = await anyio.to_thread.run_sync(
            partial(generate_image_prompt, title=payload.title, description=payload.description)
        )
    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            code="PROMPT_GENERATION_FAILED",
            message="Failed to generate image prompt",
            http_status=502,
            detail={"error": repr(exc)},
        ) from exc

    async with limit.semaphore:
        try:
            call_generate = partial(
                model.generate,
                prompt=prompt,
                height=FIXED_HEIGHT,
                width=FIXED_WIDTH,
                num_inference_steps=FIXED_STEPS,
                guidance_scale=FIXED_GUIDANCE_SCALE,
                seed=FIXED_SEED,
            )
            image = await anyio.to_thread.run_sync(call_generate)
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError(detail=str(exc)) from exc
        except Exception as exc:
            raise InferenceError(detail=str(exc)) from exc

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


async def generate_image_png_to_r2(request: Request, payload: GeneratePlayPosterToR2Request) -> str:
    r2 = getattr(request.app.state, "r2", None)
    if r2 is None:
        raise AppError(code="R2_NOT_ENABLED", message="Cloudflare R2 is not enabled", http_status=500)

    png_bytes = await generate_image_png(request, payload)

    prefix = _image_remote_prefix()
    key = payload.key or f"images/generated/{uuid.uuid4()}.png"
    key = _apply_prefix(prefix=prefix, key=key)

    # boto3 is sync; run in worker thread.
    upload = partial(
        r2.upload_bytes,
        key=key,
        data=png_bytes,
        content_type="image/png",
    )
    await anyio.to_thread.run_sync(upload)
    return key
