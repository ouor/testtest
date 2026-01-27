from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI

from app.core.concurrency.limits import ModelSemaphoreRegistry
from app.core.errors.exceptions import ModelLoadError
from app.domains.image_generation.model import ZImageTurboModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.limits = ModelSemaphoreRegistry()

    try:
        # Heavy init: do it once at startup.
        model = await anyio.to_thread.run_sync(ZImageTurboModel.load_from_env)
    except Exception as exc:
        raise ModelLoadError("Failed to load image generation model", cause=exc) from exc

    app.state.models["zimage_turbo"] = model
    app.state.limits.register("zimage_turbo", max_concurrency=1)

    yield

    # Best-effort cleanup.
    try:
        await anyio.to_thread.run_sync(model.close)
    except Exception:
        pass
