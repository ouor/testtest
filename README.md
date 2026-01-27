# AI Model Server (FastAPI)

## Run (image generation only)

```bash
conda create -n "default" python=3.10
conda activate default
conda install forge:uv
uv pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health checks:
- `GET /healthz`
- `GET /readyz`

Image generation:

```bash
curl -s http://localhost:8000/v1/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A serene landscape with mountains and a river during sunset.","seed":42}' \
  --output out.png
```

### Notes
- Model loads once at startup (FastAPI lifespan).
- Inference is serialized with a semaphore (`max_concurrency=1`).

Environment variables:
- `IMAGE_ENABLED` (default: `1`) - set `0` to skip image model loading
- `IMAGE_MODEL_NAME` (default: `Tongyi-MAI/Z-Image-Turbo`)

## Voice generation (optional)

Enable voice model loading at startup:

```bash
export VOICE_ENABLED=1
export VOICE_MODEL_NAME="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
```

Generate voice (multipart form upload):

```bash
curl -s http://localhost:8000/v1/voice/generate \
  -F "ref_audio=@test/ref.mp3" \
  -F "ref_text=아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든.. 요! 근데 그 쪽에서 자꾸 안 믿어주니까!" \
  -F "text=오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?" \
  -F "language=Korean" \
  --output out.mp3
```

Notes:
- Endpoint returns `audio/mpeg` bytes (mp3).
- MP3 encoding uses `torchaudio`; depending on your environment, an ffmpeg-backed build may be required.
