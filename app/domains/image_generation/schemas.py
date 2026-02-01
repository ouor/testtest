from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratePlayPosterRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=4000)


class GeneratePlayPosterToR2Request(GeneratePlayPosterRequest):
    key: str | None = Field(default=None, min_length=1, max_length=1024)


class R2KeyResponse(BaseModel):
    key: str
