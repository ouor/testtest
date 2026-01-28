from __future__ import annotations

import anyio
import torch
from fastapi import Request, UploadFile

from app.core.errors.exceptions import AppError, InferenceError, ModelLoadError, OutOfMemoryError
from app.domains.image_search.model import IMAGE_SEARCH_KEY, ImageRecord
from app.domains.image_search.schemas import SearchImagesRequest


def _get_state(request: Request):
    state = getattr(request.app.state, IMAGE_SEARCH_KEY, None)
    if state is None:
        raise ModelLoadError("Image search is not initialized")
    return state


async def register_image(request: Request, *, file: UploadFile) -> ImageRecord:
    state = _get_state(request)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise AppError(code="INVALID_IMAGE", message="Only image/* uploads are allowed", http_status=400)

    raw = await file.read()
    if not raw:
        raise AppError(code="EMPTY_FILE", message="Uploaded file is empty", http_status=400)

    max_bytes = int((__import__("os").getenv("IMAGE_SEARCH_MAX_BYTES", str(20 * 1024 * 1024))))
    if len(raw) > max_bytes:
        raise AppError(code="FILE_TOO_LARGE", message="Uploaded file is too large", http_status=413)

    image_id = state.new_id()

    from app.domains.image_search.model import _safe_suffix

    suffix = _safe_suffix(file.filename)
    path = state.blob_store.save(image_id=image_id, data=raw, suffix=suffix)

    # Embed in a worker thread to avoid blocking the event loop.
    try:
        limit = request.app.state.limits.get(IMAGE_SEARCH_KEY)
    except Exception:
        limit = None

    try:
        if limit is None:
            vec = await anyio.to_thread.run_sync(state.embedder.embed_image_path, str(path))
        else:
            async with limit.semaphore:
                vec = await anyio.to_thread.run_sync(state.embedder.embed_image_path, str(path))
    except torch.cuda.OutOfMemoryError as exc:
        state.blob_store.delete(path=path)
        raise OutOfMemoryError(detail=str(exc)) from exc
    except Exception as exc:
        state.blob_store.delete(path=path)
        raise InferenceError(detail=str(exc)) from exc

    record = ImageRecord(
        id=image_id,
        path=path,
        content_type=file.content_type,
        original_filename=file.filename,
        size_bytes=len(raw),
    )

    async with state.lock:
        state.records[image_id] = record
        state.vector_index.upsert(item_id=image_id, vector=vec)

    return record


async def delete_image(request: Request, *, image_id: str) -> None:
    state = _get_state(request)
    async with state.lock:
        record = state.records.pop(image_id, None)
        state.vector_index.delete(item_id=image_id)

    if record is None:
        raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)

    state.blob_store.delete(path=record.path)


async def list_images(request: Request) -> list[ImageRecord]:
    state = _get_state(request)
    async with state.lock:
        records = list(state.records.values())
    records.sort(key=lambda r: r.id)
    return records


async def search_images(request: Request, *, payload: SearchImagesRequest) -> list[tuple[str, float]]:
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
        return state.vector_index.search(vector=vec, limit=payload.limit)


async def get_image_record(request: Request, *, image_id: str) -> ImageRecord:
    state = _get_state(request)
    async with state.lock:
        record = state.records.get(image_id)
    if record is None:
        raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)
    return record
