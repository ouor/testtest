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


def _validate_uuid(image_id: str) -> str:
    try:
        uuid.UUID(image_id)
    except Exception:
        raise AppError(code="INVALID_ID", message="Invalid image id", http_status=400)
    return image_id


@router.post("/images", response_model=UploadImageResponse)
async def upload_image_endpoint(request: Request, file: UploadFile = File(...)):
    record = await register_image(request, file=file)
    if not record.r2_key:
        raise AppError(code="R2_KEY_MISSING", message="Failed to persist image to R2", http_status=500)
    return UploadImageResponse(
        id=record.id,
        r2_key=record.r2_key,
        original_filename=record.original_filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
    )


@router.delete("/images/{image_id}", status_code=204)
async def delete_image_endpoint(request: Request, image_id: str):
    image_id = _validate_uuid(image_id)
    await delete_image(request, image_id=image_id)
    return None


@router.get("/images", response_model=ListImagesResponse)
async def list_images_endpoint(request: Request):
    records = await list_images(request)
    return ListImagesResponse(
        images=[
            ImageInfo(
                id=r.id,
                r2_key=r.r2_key,
                original_filename=r.original_filename,
                content_type=r.content_type,
                size_bytes=r.size_bytes,
            )
            for r in records
        ]
    )


@router.post("/images/search", response_model=SearchImagesResponse)
async def search_images_endpoint(request: Request, payload: SearchImagesRequest):
    results = await search_images(request, payload=payload)
    return SearchImagesResponse(
        results=[
            SearchResult(
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
    "/images/{image_id}/file",
    responses={200: {"content": {"image/*": {}}}},
)
async def get_image_file_endpoint(request: Request, image_id: str):
    image_id = _validate_uuid(image_id)
    record = await get_image_record(request, image_id=image_id)
    url = await get_image_presigned_url(request, image_id=image_id)
    return RedirectResponse(url=url, status_code=307)
