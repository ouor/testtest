from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class ModelLimit:
    semaphore: asyncio.Semaphore


class ModelSemaphoreRegistry:
    def __init__(self) -> None:
        self._limits: dict[str, ModelLimit] = {}

    def register(self, name: str, *, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._limits[name] = ModelLimit(semaphore=asyncio.Semaphore(max_concurrency))

    def get(self, name: str) -> ModelLimit:
        return self._limits[name]
