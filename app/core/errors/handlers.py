from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors.exceptions import AppError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception):
        # Avoid leaking internal details; log separately if needed.
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL", "message": "Internal server error"}},
        )
