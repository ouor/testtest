from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AppError(Exception):
    code: str
    message: str
    http_status: int = 400
    detail: Any | None = None
    cause: Exception | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.detail is not None:
            payload["error"]["detail"] = self.detail
        return payload


class ModelLoadError(AppError):
    def __init__(self, message: str = "Model load failed", *, cause: Exception | None = None):
        super().__init__(code="MODEL_LOAD_FAILED", message=message, http_status=500, cause=cause)


class InferenceError(AppError):
    def __init__(self, message: str = "Inference failed", *, detail: Any | None = None):
        super().__init__(code="INFERENCE_FAILED", message=message, http_status=500, detail=detail)


class OutOfMemoryError(AppError):
    def __init__(self, message: str = "Out of memory", *, detail: Any | None = None):
        super().__init__(code="OUT_OF_MEMORY", message=message, http_status=503, detail=detail)
