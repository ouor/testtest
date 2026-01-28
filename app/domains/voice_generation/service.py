from __future__ import annotations

import os
import tempfile
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import torch
import torchaudio
from fastapi import Request, UploadFile

from app.core.errors.exceptions import InferenceError, ModelLoadError, OutOfMemoryError
from app.core.errors.exceptions import AppError
from app.domains.voice_generation.model import VOICE_MODEL_KEY
from app.domains.voice_generation.schemas import GenerateVoiceRequest, GenerateVoiceToR2Request

def _safe_suffix(filename: str | None) -> str:
    if not filename:
        return ".bin"
    suffix = Path(filename).suffix
    if not suffix or len(suffix) > 16:
        return ".bin"
    return suffix


def _to_mono_tensor(wav: Any) -> torch.Tensor:
    # qwen-tts returns (wav, sr) where wav is typically a list/array; demo uses wav[0].
    if isinstance(wav, (list, tuple)):
        wav = wav[0]

    tensor = torch.as_tensor(wav)
    if tensor.ndim == 2:
        # If (channels, time) pick first channel.
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(f"Unexpected wav shape: {tuple(tensor.shape)}")

    return tensor.to(dtype=torch.float32).unsqueeze(0)  # (1, time)


def _encode_mp3_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    # torchaudio.save to mp3 relies on the ffmpeg backend in many builds.
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torchaudio.save(tmp_path, waveform.cpu(), sample_rate, format="mp3")
        return Path(tmp_path).read_bytes()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


async def generate_voice_mp3(
    request: Request,
    *,
    ref_audio: UploadFile,
    payload: GenerateVoiceRequest,
) -> bytes:
    model = request.app.state.models.get(VOICE_MODEL_KEY)
    if model is None:
        raise ModelLoadError("Voice model is not loaded")

    try:
        limit = request.app.state.limits.get(VOICE_MODEL_KEY)
    except Exception as exc:
        raise ModelLoadError("Concurrency limits not initialized") from exc

    # Persist uploaded audio to a temp file because qwen-tts expects a file path.
    suffix = _safe_suffix(ref_audio.filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_ref:
        ref_path = tmp_ref.name

    try:
        data = await ref_audio.read()
        await anyio.to_thread.run_sync(Path(ref_path).write_bytes, data)

        async with limit.semaphore:
            try:
                make_prompt = partial(
                    model.create_voice_clone_prompt,
                    ref_audio_path=ref_path,
                    ref_text=payload.ref_text,
                )
                prompt = await anyio.to_thread.run_sync(make_prompt)

                do_generate = partial(
                    model.generate_voice_clone,
                    text=payload.text,
                    language=payload.language,
                    voice_clone_prompt=prompt,
                )
                wav, sr = await anyio.to_thread.run_sync(do_generate)

                waveform = _to_mono_tensor(wav)
                mp3_bytes = await anyio.to_thread.run_sync(_encode_mp3_bytes, waveform, int(sr))
                return mp3_bytes
            except torch.cuda.OutOfMemoryError as exc:
                raise OutOfMemoryError(detail=str(exc)) from exc
            except Exception as exc:
                raise InferenceError(detail=str(exc)) from exc
    finally:
        try:
            os.remove(ref_path)
        except Exception:
            pass


async def generate_voice_mp3_to_r2(request: Request, *, payload: GenerateVoiceToR2Request) -> str:
    r2 = getattr(request.app.state, "r2", None)
    if r2 is None:
        raise AppError(code="R2_NOT_ENABLED", message="Cloudflare R2 is not enabled", http_status=500)

    # Download reference audio from R2 to a temp file, since qwen-tts expects a file path.
    suffix = Path(payload.ref_audio_key).suffix
    if not suffix or len(suffix) > 16:
        suffix = ".bin"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_ref:
        ref_path = tmp_ref.name

    try:
        # boto3 is sync; run in worker thread.
        ref_bytes = await anyio.to_thread.run_sync(partial(r2.download_bytes, key=payload.ref_audio_key))
        await anyio.to_thread.run_sync(Path(ref_path).write_bytes, ref_bytes)

        # Reuse existing generation pipeline.
        class _UploadLike:
            filename = None

            async def read(self):
                return ref_bytes

        # We can't pass UploadFile easily without Starlette internals; call the core logic inline.
        model = request.app.state.models.get(VOICE_MODEL_KEY)
        if model is None:
            raise ModelLoadError("Voice model is not loaded")

        try:
            limit = request.app.state.limits.get(VOICE_MODEL_KEY)
        except Exception as exc:
            raise ModelLoadError("Concurrency limits not initialized") from exc

        async with limit.semaphore:
            try:
                make_prompt = partial(
                    model.create_voice_clone_prompt,
                    ref_audio_path=ref_path,
                    ref_text=payload.ref_text,
                )
                prompt = await anyio.to_thread.run_sync(make_prompt)

                do_generate = partial(
                    model.generate_voice_clone,
                    text=payload.text,
                    language=payload.language,
                    voice_clone_prompt=prompt,
                )
                wav, sr = await anyio.to_thread.run_sync(do_generate)

                waveform = _to_mono_tensor(wav)
                mp3_bytes = await anyio.to_thread.run_sync(_encode_mp3_bytes, waveform, int(sr))
            except torch.cuda.OutOfMemoryError as exc:
                raise OutOfMemoryError(detail=str(exc)) from exc
            except Exception as exc:
                raise InferenceError(detail=str(exc)) from exc

        out_key = payload.key or f"voice/generated/{uuid.uuid4()}.mp3"
        await anyio.to_thread.run_sync(
            partial(
                r2.upload_bytes,
                key=out_key,
                data=mp3_bytes,
                content_type="audio/mpeg",
            )
        )
        return out_key
    finally:
        try:
            os.remove(ref_path)
        except Exception:
            pass
