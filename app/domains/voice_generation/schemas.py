from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateVoiceRequest(BaseModel):
    ref_text: str = Field(..., min_length=1, max_length=2000)
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(..., min_length=2, max_length=10)


class GenerateVoiceToR2Request(GenerateVoiceRequest):
    ref_audio_key: str = Field(..., min_length=1, max_length=1024)
    key: str | None = Field(default=None, min_length=1, max_length=1024)


class R2KeyResponse(BaseModel):
    key: str
