from __future__ import annotations

from pydantic import BaseModel, Field


class UploadImageResponse(BaseModel):
    id: str
    original_filename: str | None = None
    content_type: str
    size_bytes: int


class ImageInfo(BaseModel):
    id: str
    original_filename: str | None = None
    content_type: str
    size_bytes: int


class ListImagesResponse(BaseModel):
    images: list[ImageInfo]


class SearchImagesRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=100)


class SearchResult(BaseModel):
    id: str
    score: float


class SearchImagesResponse(BaseModel):
    results: list[SearchResult]
