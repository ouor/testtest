from __future__ import annotations

from contextlib import asynccontextmanager

import os
import tempfile
from functools import partial
from pathlib import Path

import anyio
from fastapi import FastAPI

from app.core.concurrency.limits import ModelSemaphoreRegistry
from app.core.errors.exceptions import ModelLoadError
from app.domains.image_generation.model import ZImageTurboModel


def _env_truthy(name: str, default: str = "0") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: str) -> int:
    raw = (os.getenv(name, default) or "").strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


async def _image_search_restore_db_from_r2(app: FastAPI) -> None:
    r2 = getattr(app.state, "r2", None)
    if r2 is None:
        return

    r2_key = (os.getenv("IMAGE_SEARCH_DB_R2_KEY") or "image_search.db").strip()
    if not r2_key:
        r2_key = "image_search.db"

    from app.domains.image_search.model import resolve_db_path_from_env, resolve_project_root

    project_root = resolve_project_root()
    db_path = resolve_db_path_from_env(project_root=project_root)

    app.state.image_search_db_path = db_path
    app.state.image_search_db_r2_key = r2_key

    # Only download when the local DB does not exist.
    if db_path.exists() and db_path.stat().st_size > 0:
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temp file first, then atomically replace.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await anyio.to_thread.run_sync(partial(r2.download_file, key=r2_key, path=tmp_path))
    except Exception:
        # If not found or download fails, just start with a fresh DB.
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return

    await anyio.to_thread.run_sync(partial(Path(tmp_path).replace, db_path))
    print(f"Restored image search DB from R2: key={r2_key} -> {db_path}")


async def _image_search_backup_db_to_r2(app: FastAPI) -> None:
    r2 = getattr(app.state, "r2", None)
    if r2 is None:
        return

    state = getattr(app.state, "image_search", None)
    if state is None:
        return

    r2_key = getattr(app.state, "image_search_db_r2_key", None) or (os.getenv("IMAGE_SEARCH_DB_R2_KEY") or "").strip()
    if not r2_key:
        r2_key = "image_search.db"

    db_path = getattr(app.state, "image_search_db_path", None)
    if not isinstance(db_path, Path):
        from app.domains.image_search.model import resolve_db_path_from_env, resolve_project_root

        db_path = resolve_db_path_from_env(project_root=resolve_project_root())

    # Snapshot to temp file to avoid uploading a partially-written DB.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        snapshot_path = Path(tmp.name)

    try:
        async with state.lock:
            await anyio.to_thread.run_sync(partial(state.vector_index.backup_to_path, dest_path=snapshot_path))
        await anyio.to_thread.run_sync(partial(r2.upload_file, path=str(snapshot_path), key=r2_key))
        print(f"Backed up image search DB to R2: {snapshot_path} -> key={r2_key}")
    finally:
        try:
            snapshot_path.unlink(missing_ok=True)
        except Exception:
            pass


async def _image_search_periodic_backup_task(app: FastAPI) -> None:
    interval = _env_int("IMAGE_SEARCH_DB_BACKUP_INTERVAL_SECONDS", "1800")
    if interval < 60:
        interval = 60
    while True:
        await anyio.sleep(float(interval))
        try:
            await _image_search_backup_db_to_r2(app)
        except Exception:
            # Best-effort periodic backup.
            pass


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
    app.state.image_search_backup_tg = None
    try:
        from app.domains.image_search.model import IMAGE_SEARCH_KEY, create_state_from_env, enabled_from_env

        if enabled_from_env():
            # If configured, restore the SQLite DB from R2 before initializing vectorlite.
            await _image_search_restore_db_from_r2(app)

            # Heavy init: do it once at startup.
            image_search_state = await anyio.to_thread.run_sync(create_state_from_env)
            app.state.image_search = image_search_state
            app.state.limits.register(IMAGE_SEARCH_KEY, max_concurrency=1)
            print("Initialized image search")

            # Start periodic DB backups when R2 is available.
            if getattr(app.state, "r2", None) is not None and _env_truthy("IMAGE_SEARCH_DB_BACKUP_ENABLED", "1"):
                tg = anyio.create_task_group()
                await tg.__aenter__()
                app.state.image_search_backup_tg = tg
                tg.start_soon(_image_search_periodic_backup_task, app)
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

    # Best-effort final backup of image search DB.
    try:
        tg = getattr(app.state, "image_search_backup_tg", None)
        if tg is not None:
            tg.cancel_scope.cancel()
            await tg.__aexit__(None, None, None)
    except Exception:
        pass

    try:
        if getattr(app.state, "image_search", None) is not None and getattr(app.state, "r2", None) is not None:
            if _env_truthy("IMAGE_SEARCH_DB_BACKUP_ENABLED", "1"):
                await _image_search_backup_db_to_r2(app)
    except Exception:
        pass

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
