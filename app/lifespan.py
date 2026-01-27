from __future__ import annotations

from contextlib import asynccontextmanager

import os
import anyio
from fastapi import FastAPI

from app.core.concurrency.limits import ModelSemaphoreRegistry
from app.core.errors.exceptions import ModelLoadError
from app.domains.image_generation.model import ZImageTurboModel


def _env_truthy(name: str, default: str = "0") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.limits = ModelSemaphoreRegistry()

    # Optional image model (enabled by default).
    if _env_truthy("IMAGE_ENABLED", "1"):
        try:
            # Heavy init: do it once at startup.
            image_model = await anyio.to_thread.run_sync(ZImageTurboModel.load_from_env)
            app.state.models["zimage_turbo"] = image_model
            app.state.limits.register("zimage_turbo", max_concurrency=1)
            print("Loaded Z-Image-Turbo model")
        except Exception as exc:
            raise ModelLoadError("Failed to load image generation model", cause=exc) from exc

    # Optional voice model.
    if _env_truthy("VOICE_ENABLED", "0"):
        try:
            from app.domains.voice_generation.model import Qwen3TTSModelWrapper, VOICE_MODEL_KEY

            voice_model = await anyio.to_thread.run_sync(Qwen3TTSModelWrapper.load_from_env)
            app.state.models[VOICE_MODEL_KEY] = voice_model
            app.state.limits.register(VOICE_MODEL_KEY, max_concurrency=1)
            print("Loaded Qwen3-TTS model")
        except Exception as exc:
            raise ModelLoadError("Failed to load voice generation model", cause=exc) from exc

    yield

    # Best-effort cleanup.
    image_model = app.state.models.get("zimage_turbo")
    if image_model is not None:
        try:
            await anyio.to_thread.run_sync(image_model.close)
        except Exception:
            pass

    voice_key = "qwen3_tts"
    try:
        from app.domains.voice_generation.model import VOICE_MODEL_KEY

        voice_key = VOICE_MODEL_KEY
    except Exception:
        pass

    voice_model = app.state.models.get(voice_key)
    if voice_model is not None:
        try:
            await anyio.to_thread.run_sync(voice_model.close)
        except Exception:
            pass
