from __future__ import annotations

from pydantic import BaseModel, Field


class UploadImageResponse(BaseModel):
    project_id: str
    id: str
    r2_key: str
    original_filename: str | None = None
    content_type: str
    size_bytes: int


class ImageInfo(BaseModel):
    project_id: str
    id: str
    r2_key: str
    original_filename: str | None = None
    content_type: str
    size_bytes: int


class ListImagesResponse(BaseModel):
    images: list[ImageInfo]


class SearchImagesRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=100)


class SearchResult(BaseModel):
    project_id: str
    id: str
    r2_key: str
    score: float
    original_filename: str | None = None
    content_type: str
    size_bytes: int


class SearchImagesResponse(BaseModel):
    results: list[SearchResult]
