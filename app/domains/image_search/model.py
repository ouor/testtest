from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.domains.image_search.vectordb import ImageRecordStore, VectorliteVectorIndex

IMAGE_SEARCH_KEY = "image_search"


def _env_truthy(name: str, default: str = "0") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _safe_suffix(filename: str | None) -> str:
    if not filename:
        return ".bin"
    suffix = Path(filename).suffix
    if not suffix or len(suffix) > 16:
        return ".bin"
    return suffix


class ClipEmbedder:
    def __init__(self, *, model_name: str, device: str) -> None:
        self.device = device
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @classmethod
    def load_from_env(cls) -> "ClipEmbedder":
        model_name = "Bingsu/clip-vit-large-patch14-ko"
        device = "cuda"

        if not torch.cuda.is_available():
            raise RuntimeError("IMAGE_SEARCH requires CUDA but torch.cuda.is_available() is False")

        return cls(model_name=model_name, device=device)

    def embed_image_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        for k in list(inputs.keys()):
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(self.device)

        with torch.inference_mode():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0].detach().cpu().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        for k in list(inputs.keys()):
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(self.device)

        with torch.inference_mode():
            outputs = self.model.get_text_features(**inputs)
        return outputs[0].detach().cpu().numpy()

    def close(self) -> None:
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


@dataclass
class ImageSearchState:
    vector_index: VectorliteVectorIndex
    record_store: ImageRecordStore
    embedder: ClipEmbedder
    lock: asyncio.Lock

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def close(self) -> None:
        try:
            self.vector_index.close()
        except Exception:
            pass

        try:
            if self.record_store is not self.vector_index:
                close_fn = getattr(self.record_store, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            pass

        self.embedder.close()


def create_state_from_env() -> ImageSearchState:
    project_root = resolve_project_root()

    db_path = resolve_db_path_from_env(project_root=project_root)

    embedder = ClipEmbedder.load_from_env()

    max_elements_raw = (os.getenv("IMAGE_SEARCH_MAX_ELEMENTS") or "50000").strip()
    try:
        max_elements = int(max_elements_raw)
    except Exception as exc:
        raise RuntimeError("IMAGE_SEARCH_MAX_ELEMENTS must be an integer") from exc
    if max_elements <= 0:
        raise RuntimeError("IMAGE_SEARCH_MAX_ELEMENTS must be > 0")
    # Derive embedding dim from the model.
    dim_probe = embedder.embed_text("dim")
    vector_dim = int(dim_probe.shape[0])

    vector_index = VectorliteVectorIndex(
        db_path=db_path,
        vector_dim=vector_dim,
        max_elements=max_elements,
    )
    return ImageSearchState(vector_index=vector_index, record_store=vector_index, embedder=embedder, lock=asyncio.Lock())


def enabled_from_env() -> bool:
    return _env_truthy("IMAGE_SEARCH_ENABLED", "1")


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_db_path_from_env(*, project_root: Path | None = None) -> Path:
    project_root = project_root or resolve_project_root()
    app_dir = project_root / "app"

    db_path_raw = (os.getenv("IMAGE_SEARCH_DB_PATH") or "").strip()
    if db_path_raw:
        db_path = Path(db_path_raw)
        if not db_path.is_absolute():
            db_path = project_root / db_path
    else:
        db_path = app_dir / "image_search.db"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


