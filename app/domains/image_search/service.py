from __future__ import annotations

import os
import tempfile
from functools import partial
from pathlib import Path

import anyio
import torch
from fastapi import Request, UploadFile

from app.core.errors.exceptions import AppError, InferenceError, ModelLoadError, OutOfMemoryError
from app.domains.image_search.model import IMAGE_SEARCH_KEY, _safe_suffix
from app.domains.image_search.vectordb import ImageRecord
from app.domains.image_search.schemas import SearchImagesRequest


def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip()
    if not prefix:
        return ""
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _image_search_remote_prefix() -> str:
    # For R2 keys of uploaded images.
    # Example: AI/SEARCH/{project_id}/{image_id}.jpg
    return _normalize_prefix(os.getenv("IMAGE_SEARCH_REMOTE_PREFIX", "AI/SEARCH/"))


def _validate_project_id(project_id: str) -> str:
    project_id = (project_id or "").strip()
    if not project_id:
        raise AppError(code="INVALID_PROJECT", message="project_id is required", http_status=400)
    if len(project_id) > 128:
        raise AppError(code="INVALID_PROJECT", message="project_id is too long", http_status=400)
    import re

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", project_id):
        raise AppError(code="INVALID_PROJECT", message="Invalid project_id format", http_status=400)
    return project_id


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


async def _best_effort_r2_delete(*, r2: object, key: str, reason: str) -> None:
    try:
        await anyio.to_thread.run_sync(partial(getattr(r2, "delete"), key=key))
    except Exception as exc:
        # Best-effort cleanup; log for observability.
        print(f"WARNING: failed to delete R2 object: key={key} reason={reason} error={exc!r}")


async def register_image(request: Request, *, project_id: str, file: UploadFile) -> ImageRecord:
    state = _get_state(request)
    r2 = _get_r2(request)
    project_id = _validate_project_id(project_id)

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

    # Upload original bytes to R2 (project-scoped).
    r2_prefix = _image_search_remote_prefix()
    r2_key = f"{r2_prefix}{project_id}/{image_id}{suffix}"
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
            await _best_effort_r2_delete(r2=r2, key=r2_key, reason="embedding_oom")
            raise OutOfMemoryError(detail=str(exc)) from exc
        except Exception as exc:
            await _best_effort_r2_delete(r2=r2, key=r2_key, reason="embedding_error")
            raise InferenceError(detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    record = ImageRecord(
        project_id=project_id,
        id=image_id,
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
        await _best_effort_r2_delete(r2=r2, key=r2_key, reason="db_write_failed")
        raise InferenceError(detail=str(exc)) from exc

    return record


async def delete_image(request: Request, *, project_id: str, image_id: str) -> None:
    state = _get_state(request)
    r2 = _get_r2(request)
    project_id = _validate_project_id(project_id)
    async with state.lock:
        exists = getattr(state.record_store, "project_exists", None)
        if callable(exists) and not exists(project_id=project_id):
            raise AppError(code="PROJECT_NOT_FOUND", message="Project not found", http_status=404)
        record = state.record_store.get_record(project_id=project_id, image_id=image_id)
        if record is not None:
            state.vector_index.delete_image(project_id=project_id, image_id=image_id)

    if record is None:
        raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)

    if not record.r2_key:
        raise AppError(code="R2_KEY_MISSING", message="Image record is missing r2_key", http_status=500)

    try:
        await anyio.to_thread.run_sync(partial(getattr(r2, "delete"), key=record.r2_key))
    except Exception as exc:
        raise AppError(
            code="R2_DELETE_FAILED",
            message="Failed to delete image blob from R2",
            http_status=502,
            detail={"key": record.r2_key, "error": repr(exc)},
        ) from exc


async def list_images(request: Request, *, project_id: str) -> list[ImageRecord]:
    state = _get_state(request)
    project_id = _validate_project_id(project_id)
    async with state.lock:
        exists = getattr(state.record_store, "project_exists", None)
        if callable(exists) and not exists(project_id=project_id):
            raise AppError(code="PROJECT_NOT_FOUND", message="Project not found", http_status=404)
        return state.record_store.list_records(project_id=project_id)


async def search_images(
    request: Request, *, project_id: str, payload: SearchImagesRequest
) -> list[tuple[ImageRecord, float]]:
    state = _get_state(request)
    project_id = _validate_project_id(project_id)

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
        exists = getattr(state.record_store, "project_exists", None)
        if callable(exists) and not exists(project_id=project_id):
            raise AppError(code="PROJECT_NOT_FOUND", message="Project not found", http_status=404)
        # Prefer a single DB query that also returns metadata.
        search_records = getattr(state.vector_index, "search_records", None)
        if callable(search_records):
            return search_records(project_id=project_id, vector=vec, limit=payload.limit)

        # Fallback: resolve ids to records (older interface).
        ids = state.vector_index.search(vector=vec, limit=payload.limit)
        out: list[tuple[ImageRecord, float]] = []
        for image_id, score in ids:
            rec = state.record_store.get_record(project_id=project_id, image_id=image_id)
            if rec is not None:
                out.append((rec, score))
        return out


async def get_image_record(request: Request, *, project_id: str, image_id: str) -> ImageRecord:
    state = _get_state(request)
    project_id = _validate_project_id(project_id)
    async with state.lock:
        exists = getattr(state.record_store, "project_exists", None)
        if callable(exists) and not exists(project_id=project_id):
            raise AppError(code="PROJECT_NOT_FOUND", message="Project not found", http_status=404)
        record = state.record_store.get_record(project_id=project_id, image_id=image_id)
        if record is None:
            raise AppError(code="NOT_FOUND", message="Image not found", http_status=404)

        if not record.r2_key:
            raise AppError(code="R2_KEY_MISSING", message="Image record is missing r2_key", http_status=500)

        return record


async def get_image_presigned_url(
    request: Request, *, project_id: str, image_id: str, expires_in: int = 3600
) -> str:
    state = _get_state(request)
    r2 = _get_r2(request)
    project_id = _validate_project_id(project_id)

    async with state.lock:
        exists = getattr(state.record_store, "project_exists", None)
        if callable(exists) and not exists(project_id=project_id):
            raise AppError(code="PROJECT_NOT_FOUND", message="Project not found", http_status=404)
        record = state.record_store.get_record(project_id=project_id, image_id=image_id)
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
