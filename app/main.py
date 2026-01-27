from pathlib import Path

from fastapi import FastAPI

# Load project-root .env if present.
# Note: Uvicorn does not automatically load it unless started with --env-file.
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        print(
            "WARNING: .env file exists but python-dotenv is not installed. "
            "Install python-dotenv or start uvicorn with --env-file .env"
        )
    else:
        # Prefer .env values over inherited shell env vars for local runs.
        load_dotenv(dotenv_path=_env_path, override=True)

from app.core.errors.handlers import register_exception_handlers
from app.domains.image_generation.router import router as image_router
from app.domains.voice_generation.router import router as voice_router
from app.lifespan import lifespan


def create_app() -> FastAPI:
    app = FastAPI(title="AI Model Server", version="0.1.0", lifespan=lifespan)

    register_exception_handlers(app)

    app.include_router(image_router, prefix="/v1")
    app.include_router(voice_router, prefix="/v1")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        models = getattr(app.state, "models", None)
        model_status = {
            "zimage_turbo": bool(models and models.get("zimage_turbo")),
            "qwen3_tts": bool(models and models.get("qwen3_tts")),
        }
        return {"ready": any(model_status.values()), "models": model_status}

    return app


app = create_app()
