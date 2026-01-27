from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    height: int = Field(default=1024, ge=64, le=2048)
    width: int = Field(default=1024, ge=64, le=2048)
    num_inference_steps: int = Field(default=9, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
