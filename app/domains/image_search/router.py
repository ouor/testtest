from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import RedirectResponse

from app.core.errors.exceptions import AppError
from app.domains.image_search.schemas import (
    ImageInfo,
    ListImagesResponse,
    SearchImagesRequest,
    SearchImagesResponse,
    SearchResult,
    UploadImageResponse,
)
from app.domains.image_search.service import (
    delete_image,
    get_image_record,
    get_image_presigned_url,
    list_images,
    register_image,
    search_images,
)

router = APIRouter(tags=["image-search"])


def _validate_project_id(project_id: str) -> str:
    project_id = (project_id or "").strip()
    if not project_id:
        raise AppError(code="INVALID_PROJECT", message="project_id is required", http_status=400)
    if len(project_id) > 128:
        raise AppError(code="INVALID_PROJECT", message="project_id is too long", http_status=400)
    # Keep it URL/DB friendly.
    import re

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", project_id):
        raise AppError(code="INVALID_PROJECT", message="Invalid project_id format", http_status=400)
    return project_id


def _validate_uuid(image_id: str) -> str:
    try:
        uuid.UUID(image_id)
    except Exception:
        raise AppError(code="INVALID_ID", message="Invalid image id", http_status=400)
    return image_id


@router.post("/projects/{project_id}/images", response_model=UploadImageResponse)
async def upload_image_endpoint(request: Request, project_id: str, file: UploadFile = File(...)):
    project_id = _validate_project_id(project_id)
    record = await register_image(request, project_id=project_id, file=file)
    if not record.r2_key:
        raise AppError(code="R2_KEY_MISSING", message="Failed to persist image to R2", http_status=500)
    return UploadImageResponse(
        project_id=record.project_id,
        id=record.id,
        r2_key=record.r2_key,
        original_filename=record.original_filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
    )


@router.delete("/projects/{project_id}/images/{image_id}", status_code=204)
async def delete_image_endpoint(request: Request, project_id: str, image_id: str):
    project_id = _validate_project_id(project_id)
    image_id = _validate_uuid(image_id)
    await delete_image(request, project_id=project_id, image_id=image_id)
    return None


@router.get("/projects/{project_id}/images", response_model=ListImagesResponse)
async def list_images_endpoint(request: Request, project_id: str):
    project_id = _validate_project_id(project_id)
    records = await list_images(request, project_id=project_id)
    return ListImagesResponse(
        images=[
            ImageInfo(
                project_id=r.project_id,
                id=r.id,
                r2_key=r.r2_key,
                original_filename=r.original_filename,
                content_type=r.content_type,
                size_bytes=r.size_bytes,
            )
            for r in records
        ]
    )


@router.post("/projects/{project_id}/images/search", response_model=SearchImagesResponse)
async def search_images_endpoint(request: Request, project_id: str, payload: SearchImagesRequest):
    project_id = _validate_project_id(project_id)
    results = await search_images(request, project_id=project_id, payload=payload)
    return SearchImagesResponse(
        results=[
            SearchResult(
                project_id=r.project_id,
                id=r.id,
                r2_key=r.r2_key,
                score=s,
                original_filename=r.original_filename,
                content_type=r.content_type,
                size_bytes=r.size_bytes,
            )
            for r, s in results
        ]
    )


@router.get(
    "/projects/{project_id}/images/{image_id}/file",
    responses={200: {"content": {"image/*": {}}}},
)
async def get_image_file_endpoint(request: Request, project_id: str, image_id: str):
    project_id = _validate_project_id(project_id)
    image_id = _validate_uuid(image_id)
    record = await get_image_record(request, project_id=project_id, image_id=image_id)
    url = await get_image_presigned_url(request, project_id=project_id, image_id=image_id)
    return RedirectResponse(url=url, status_code=307)
