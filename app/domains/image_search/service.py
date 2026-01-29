from __future__ import annotations

import os
import tempfile
from functools import partial
from pathlib import Path

import anyio
import torch
from fastapi import Request, UploadFile

from app.core.errors.exceptions import AppError, InferenceError, ModelLoadError, OutOfMemoryError
from app.domains.image_search.model import IMAGE_SEARCH_KEY, ImageRecord, _safe_suffix
from app.domains.image_search.schemas import SearchImagesRequest


def _get_state(request: Request):
    state = getattr(request.app.state, IMAGE_SEARCH_KEY, None)
    if state is None:
        raise ModelLoadError("Image search is not initialized")
    return state


def _get_r2(request: Request):
    r2 = getattr(request.app.state, "r2", None)
    if r2 is None:
        raise AppError(code="R2_NOT_ENABLED", message="Cloudflare R2 is not enabled", http_status=500)
    return r2


async def register_image(request: Request, *, file: UploadFile) -> ImageRecord:
    state = _get_state(request)
    r2 = _get_r2(request)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise AppError(code="INVALID_IMAGE", message="Only image/* uploads are allowed", http_status=400)

    raw = await file.read()
    if not raw:
        raise AppError(code="EMPTY_FILE", message="Uploaded file is empty", http_status=400)

    max_bytes = int(os.getenv("IMAGE_SEARCH_MAX_BYTES", str(20 * 1024 * 1024)))
    if len(raw) > max_bytes:
        raise AppError(code="FILE_TOO_LARGE", message="Uploaded file is too large", http_status=413)

    image_id = state.new_id()

    suffix = _safe_suffix(file.filename)

    # Upload original bytes to R2.
    r2_key = f"image_search/{image_id}{suffix}"
    upload = partial(
        r2.upload_bytes,
        key=r2_key,
        data=raw,
        content_type=file.content_type,
    )
    await anyio.to_thread.run_sync(upload)

    # The embedder currently operates on a local file path; use a temp file.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await anyio.to_thread.run_sync(Path(tmp_path).write_bytes, raw)

        # Embed in a worker thread to avoid blocking the event loop.
        try:
            limit = request.app.state.limits.get(IMAGE_SEARCH_KEY)
        except Exception:
            limit = None

        try:
            if limit is None:
                vec = await anyio.to_thread.run_sync(state.embedder.embed_image_path, str(tmp_path))
            else:
                async with limit.semaphore:
                    vec = await anyio.to_thread.run_sync(state.embedder.embed_image_path, str(tmp_path))
        except torch.cuda.OutOfMemoryError as exc:
            # Best-effort cleanup of uploaded blob.
            try:
                await anyio.to_thread.run_sync(partial(r2.delete, key=r2_key))
            except Exception:
                pass
            raise OutOfMemoryError(detail=str(exc)) from exc
        except Exception as exc:
            try:
                await anyio.to_thread.run_sync(partial(r2.delete, key=r2_key))
            except Exception:
                pass
            raise InferenceError(detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    record = ImageRecord(
        id=image_id,
        path=None,
        content_type=file.content_type,
        original_filename=file.filename,
        size_bytes=len(raw),
        r2_key=r2_key,
    )

    try:
        async with state.lock:
            state.vector_index.upsert_image(record=record, vector=vec)
    except Exception as exc:
        # If DB write fails, attempt to delete the blob.
        try:
            await anyio.to_thread.run_sync(partial(r2.delete, key=r2_key))
        except Exception:
            pass
        raise InferenceError(detail=str(exc)) from exc

    return record


async def delete_image(request: Request, *, image_id: str) -> None:
    state = _get_state(request)
    async with state.lock:
        record = state.record_store.get_record(image_id=image_id)
        if record is not None:
            state.vector_index.delete_image(image_id=image_id)

    if record is None:
        raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)

    if record.r2_key:
        r2 = _get_r2(request)
        await anyio.to_thread.run_sync(partial(r2.delete, key=record.r2_key))
    elif record.path is not None:
        # Backward-compatible cleanup for legacy local records.
        try:
            await anyio.to_thread.run_sync(partial(record.path.unlink, missing_ok=True))
        except Exception:
            pass


async def list_images(request: Request) -> list[ImageRecord]:
    state = _get_state(request)
    async with state.lock:
        return state.record_store.list_records()


async def search_images(request: Request, *, payload: SearchImagesRequest) -> list[tuple[ImageRecord, float]]:
    state = _get_state(request)

    try:
        limit = request.app.state.limits.get(IMAGE_SEARCH_KEY)
    except Exception:
        limit = None

    try:
        if limit is None:
            vec = await anyio.to_thread.run_sync(state.embedder.embed_text, payload.query)
        else:
            async with limit.semaphore:
                vec = await anyio.to_thread.run_sync(state.embedder.embed_text, payload.query)
    except torch.cuda.OutOfMemoryError as exc:
        raise OutOfMemoryError(detail=str(exc)) from exc
    except Exception as exc:
        raise InferenceError(detail=str(exc)) from exc

    async with state.lock:
        # Prefer a single DB query that also returns metadata.
        search_records = getattr(state.vector_index, "search_records", None)
        if callable(search_records):
            return search_records(vector=vec, limit=payload.limit)

        # Fallback: resolve ids to records (older interface).
        ids = state.vector_index.search(vector=vec, limit=payload.limit)
        out: list[tuple[ImageRecord, float]] = []
        for image_id, score in ids:
            rec = state.record_store.get_record(image_id=image_id)
            if rec is not None:
                out.append((rec, score))
        return out


async def get_image_record(request: Request, *, image_id: str) -> ImageRecord:
    state = _get_state(request)
    async with state.lock:
        record = state.record_store.get_record(image_id=image_id)
        if record is None:
            raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)

        # Backward-compatible self-heal for legacy local records.
        if record.r2_key is None and record.path is not None and not record.path.exists():
            state.vector_index.delete_image(image_id=image_id)
            raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)

        return record


async def get_image_presigned_url(request: Request, *, image_id: str, expires_in: int = 3600) -> str:
    state = _get_state(request)
    r2 = _get_r2(request)

    async with state.lock:
        record = state.record_store.get_record(image_id=image_id)
        if record is None:
            raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)
        if not record.r2_key:
            raise AppError(
                code="R2_KEY_MISSING",
                message="Image is not stored in R2",
                http_status=409,
            )

        url = await anyio.to_thread.run_sync(partial(r2.presigned_get_url, key=record.r2_key, expires_in=expires_in))
        return url
