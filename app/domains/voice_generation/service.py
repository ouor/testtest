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
from botocore.exceptions import ClientError
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


def _r2_error_code(exc: ClientError) -> str:
    try:
        return str(exc.response.get("Error", {}).get("Code") or "")
    except Exception:
        return ""


def _is_r2_not_found(exc: ClientError) -> bool:
    code = _r2_error_code(exc)
    # S3-style "NoSuchKey" is typical; some stacks return 404-like codes.
    return code in {"NoSuchKey", "NotFound", "404"}


def _get_voice_model_and_limit(request: Request):
    model = request.app.state.models.get(VOICE_MODEL_KEY)
    if model is None:
        raise ModelLoadError("Voice model is not loaded")

    try:
        limit = request.app.state.limits.get(VOICE_MODEL_KEY)
    except Exception as exc:
        raise ModelLoadError("Concurrency limits not initialized") from exc

    return model, limit


async def _generate_voice_mp3_core(
    request: Request,
    *,
    ref_audio_path: str,
    payload: GenerateVoiceRequest,
) -> bytes:
    model, limit = _get_voice_model_and_limit(request)

    async with limit.semaphore:
        try:
            make_prompt = partial(
                model.create_voice_clone_prompt,
                ref_audio_path=ref_audio_path,
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


async def generate_voice_mp3(
    request: Request,
    *,
    ref_audio: UploadFile,
    payload: GenerateVoiceRequest,
) -> bytes:
    # Persist uploaded audio to a temp file because qwen-tts expects a file path.
    suffix = _safe_suffix(ref_audio.filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_ref:
        ref_path = tmp_ref.name

    try:
        data = await ref_audio.read()
        await anyio.to_thread.run_sync(Path(ref_path).write_bytes, data)

        return await _generate_voice_mp3_core(request, ref_audio_path=ref_path, payload=payload)
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
        try:
            ref_bytes = await anyio.to_thread.run_sync(partial(r2.download_bytes, key=payload.ref_audio_key))
        except ClientError as exc:
            if _is_r2_not_found(exc):
                raise AppError(
                    code="R2_KEY_NOT_FOUND",
                    message="Reference audio key not found in R2",
                    http_status=404,
                    detail={"key": payload.ref_audio_key, "error_code": _r2_error_code(exc)},
                ) from exc
            raise AppError(
                code="R2_DOWNLOAD_FAILED",
                message="Failed to download reference audio from R2",
                http_status=502,
                detail={"key": payload.ref_audio_key, "error_code": _r2_error_code(exc)},
            ) from exc
        await anyio.to_thread.run_sync(Path(ref_path).write_bytes, ref_bytes)

        mp3_bytes = await _generate_voice_mp3_core(request, ref_audio_path=ref_path, payload=payload)

        out_key = payload.key or f"voice/generated/{uuid.uuid4()}.mp3"
        try:
            await anyio.to_thread.run_sync(
                partial(
                    r2.upload_bytes,
                    key=out_key,
                    data=mp3_bytes,
                    content_type="audio/mpeg",
                )
            )
        except ClientError as exc:
            raise AppError(
                code="R2_UPLOAD_FAILED",
                message="Failed to upload generated audio to R2",
                http_status=502,
                detail={"key": out_key, "error_code": _r2_error_code(exc)},
            ) from exc
        return out_key
    finally:
        try:
            os.remove(ref_path)
        except Exception:
            pass
