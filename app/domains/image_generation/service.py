from __future__ import annotations

import io
from functools import partial

import anyio
import torch
from fastapi import Request

from app.core.errors.exceptions import InferenceError, ModelLoadError, OutOfMemoryError
from app.domains.image_generation.schemas import GenerateImageRequest


async def generate_image_png(request: Request, payload: GenerateImageRequest) -> bytes:
    model = request.app.state.models.get("zimage_turbo")
    if model is None:
        raise ModelLoadError("Image model is not loaded")

    try:
        limit = request.app.state.limits.get("zimage_turbo")
    except Exception as exc:
        raise ModelLoadError("Concurrency limits not initialized") from exc

    async with limit.semaphore:
        try:
            call_generate = partial(
                model.generate,
                prompt=payload.prompt,
                height=payload.height,
                width=payload.width,
                num_inference_steps=payload.num_inference_steps,
                guidance_scale=payload.guidance_scale,
                seed=payload.seed,
            )
            image = await anyio.to_thread.run_sync(call_generate)
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError(detail=str(exc)) from exc
        except Exception as exc:
            raise InferenceError(detail=str(exc)) from exc

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
