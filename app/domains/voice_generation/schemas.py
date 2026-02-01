from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateVoiceRequest(BaseModel):
    ref_text: str = Field(..., min_length=1, max_length=2000)
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(..., min_length=2, max_length=10)


class GenerateVoiceToR2Request(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        description="Used to locate reference files in R2: {VOICE_REMOTE_PREFIX}{user_id}.mp3 and .txt",
    )
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(..., min_length=2, max_length=10)
    key: str | None = Field(default=None, min_length=1, max_length=1024)


class R2KeyResponse(BaseModel):
    key: str
