from __future__ import annotations

import os
from dataclasses import dataclass
import inspect

import torch
from diffusers import ZImagePipeline

IMAGE_MODEL_KEY = "zimage_turbo"

@dataclass
class ZImageTurboModel:
    pipe: ZImagePipeline
    device: str

    @classmethod
    def load_from_env(cls) -> "ZImageTurboModel":
        model_name = "Tongyi-MAI/Z-Image-Turbo"
        device = "cuda"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        pipe = ZImagePipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)

        # Match test/generate_image.py behavior.
        pipe.transformer.set_attention_backend("flash")
        pipe.enable_model_cpu_offload()

        return cls(pipe=pipe, device=device)

    def generate(
        self,
        *,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ):
        generator = torch.Generator(self.device).manual_seed(seed)
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        return image

    def close(self) -> None:
        # Best-effort GPU memory cleanup.
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
