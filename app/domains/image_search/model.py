from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


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


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(vec) + 1e-8)
    return vec / denom


class VectorIndex(Protocol):
    def upsert(self, *, item_id: str, vector: np.ndarray) -> None: ...

    def delete(self, *, item_id: str) -> None: ...

    def search(self, *, vector: np.ndarray, limit: int) -> list[tuple[str, float]]: ...

    def ids(self) -> list[str]: ...


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}

    def upsert(self, *, item_id: str, vector: np.ndarray) -> None:
        self._vectors[item_id] = _normalize(vector)

    def delete(self, *, item_id: str) -> None:
        self._vectors.pop(item_id, None)

    def search(self, *, vector: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if not self._vectors or limit <= 0:
            return []

        q = _normalize(vector)
        ids = list(self._vectors.keys())
        mat = np.stack([self._vectors[i] for i in ids], axis=0)  # (n, d)
        scores = mat @ q  # (n,)

        k = min(limit, scores.shape[0])
        # argpartition for top-k then sort
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(ids[int(i)], float(scores[int(i)])) for i in top_idx]

    def ids(self) -> list[str]:
        return list(self._vectors.keys())


@dataclass(frozen=True)
class ImageRecord:
    id: str
    path: Path
    content_type: str
    original_filename: str | None
    size_bytes: int


class BlobStore(Protocol):
    def save(self, *, image_id: str, data: bytes, suffix: str) -> Path: ...

    def delete(self, *, path: Path) -> None: ...


class TempDirBlobStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def save(self, *, image_id: str, data: bytes, suffix: str) -> Path:
        path = self._base_dir / f"{image_id}{suffix}"
        path.write_bytes(data)
        return path

    def delete(self, *, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            # Best-effort
            pass


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
    temp_dir: tempfile.TemporaryDirectory
    blob_store: BlobStore
    vector_index: VectorIndex
    embedder: ClipEmbedder
    lock: asyncio.Lock
    records: dict[str, ImageRecord]

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def close(self) -> None:
        try:
            self.embedder.close()
        finally:
            try:
                self.temp_dir.cleanup()
            except Exception:
                pass


def create_state_from_env() -> ImageSearchState:
    temp_dir = tempfile.TemporaryDirectory(prefix="image-search-")
    base_dir = Path(temp_dir.name)

    embedder = ClipEmbedder.load_from_env()
    return ImageSearchState(
        temp_dir=temp_dir,
        blob_store=TempDirBlobStore(base_dir=base_dir),
        vector_index=InMemoryVectorIndex(),
        embedder=embedder,
        lock=asyncio.Lock(),
        records={},
    )


def enabled_from_env() -> bool:
    return _env_truthy("IMAGE_SEARCH_ENABLED", "1")
