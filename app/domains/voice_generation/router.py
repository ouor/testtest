from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response

from app.domains.voice_generation.service import generate_voice_mp3, generate_voice_mp3_to_r2
from app.domains.voice_generation.schemas import GenerateVoiceRequest, GenerateVoiceToR2Request, R2KeyResponse

router = APIRouter(tags=["voice-generation"])


@router.post(
    "/voice/generate",
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def generate_voice_endpoint(
    request: Request,
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
    text: str = Form(...),
    language: str = Form(...),
):
    payload = GenerateVoiceRequest(ref_text=ref_text, text=text, language=language)
    mp3_bytes = await generate_voice_mp3(
        request,
        ref_audio=ref_audio,
        payload=payload,
    )
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@router.post(
    "/r2/voice/generate",
    response_model=R2KeyResponse,
)
async def generate_voice_r2_endpoint(request: Request, payload: GenerateVoiceToR2Request):
    key = await generate_voice_mp3_to_r2(request, payload=payload)
    return R2KeyResponse(key=key)
