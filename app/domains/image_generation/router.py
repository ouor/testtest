from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.domains.image_generation.schemas import GenerateImageRequest
from app.domains.image_generation.service import generate_image_png

router = APIRouter(tags=["image-generation"])


@router.post(
    "/images/generate",
    responses={200: {"content": {"image/png": {}}}},
)
async def generate_images_endpoint(request: Request, payload: GenerateImageRequest):
    png_bytes = await generate_image_png(request, payload)
    return Response(content=png_bytes, media_type="image/png")
