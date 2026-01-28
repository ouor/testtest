from __future__ import annotations

import os
from dataclasses import dataclass

import torch


VOICE_MODEL_KEY = "qwen3_tts"


@dataclass
class Qwen3TTSModelWrapper:
    model: object
    device: str

    @classmethod
    def load_from_env(cls) -> "Qwen3TTSModelWrapper":
        # Lazy import so the server can still start when voice is disabled.
        from qwen_tts import Qwen3TTSModel  # type: ignore

        model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        model = Qwen3TTSModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        return cls(model=model, device="cuda")

    def create_voice_clone_prompt(self, *, ref_audio_path: str, ref_text: str):
        return self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    def generate_voice_clone(self, *, text: str, language: str, voice_clone_prompt):
        return self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )

    def close(self) -> None:
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
