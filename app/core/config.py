from __future__ import annotations

from pydantic import BaseModel


class Settings(BaseModel):
    device: str = "cuda"
    model_name: str = "Tongyi-MAI/Z-Image-Turbo"


settings = Settings()
