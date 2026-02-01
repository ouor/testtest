from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.domains.image_generation.schemas import GeneratePlayPosterRequest, GeneratePlayPosterToR2Request, R2KeyResponse
from app.domains.image_generation.service import generate_image_png, generate_image_png_to_r2

router = APIRouter(tags=["image-generation"])


@router.post(
    "/images/generate",
    responses={200: {"content": {"image/png": {}}}},
)
async def generate_images_endpoint(request: Request, payload: GeneratePlayPosterRequest):
    png_bytes = await generate_image_png(request, payload)
    return Response(content=png_bytes, media_type="image/png")


@router.post(
    "/r2/images/generate",
    response_model=R2KeyResponse,
)
async def generate_images_r2_endpoint(request: Request, payload: GeneratePlayPosterToR2Request):
    key = await generate_image_png_to_r2(request, payload)
    return R2KeyResponse(key=key)
