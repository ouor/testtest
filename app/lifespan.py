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

    # Optional Cloudflare R2 client.
    app.state.r2 = None
    try:
        from app.core.storage.r2 import R2Storage, r2_enabled_from_env

        if r2_enabled_from_env():
            app.state.r2 = await anyio.to_thread.run_sync(R2Storage.from_env)
            print("Initialized Cloudflare R2 client")
    except Exception as exc:
        # Only fail startup if explicitly enabled.
        try:
            from app.core.storage.r2 import r2_enabled_from_env

            if r2_enabled_from_env():
                raise ModelLoadError("Failed to initialize Cloudflare R2", cause=exc) from exc
        except Exception:
            raise

    # Optional image-search state (enabled by default).
    app.state.image_search = None
    try:
        from app.domains.image_search.model import IMAGE_SEARCH_KEY, create_state_from_env, enabled_from_env

        if enabled_from_env():
            # Heavy init: do it once at startup.
            image_search_state = await anyio.to_thread.run_sync(create_state_from_env)
            app.state.image_search = image_search_state
            app.state.limits.register(IMAGE_SEARCH_KEY, max_concurrency=1)
            print("Initialized image search")
    except Exception as exc:
        # Fail fast if image-search was expected to be enabled.
        try:
            from app.domains.image_search.model import enabled_from_env

            if enabled_from_env():
                raise ModelLoadError("Failed to initialize image search", cause=exc) from exc
        except Exception:
            # If we cannot even read the flag, keep server boot behavior unchanged.
            raise

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

    # No explicit R2 cleanup required.

    # Best-effort cleanup for image search.
    image_search_state = getattr(app.state, "image_search", None)
    if image_search_state is not None:
        try:
            await anyio.to_thread.run_sync(image_search_state.close)
        except Exception:
            pass

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
