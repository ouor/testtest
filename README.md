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

## Image search (upload + semantic search)

This feature lets you upload images, keep them in a server-side temporary directory, and search by text using a CLIP-style embedding model.

Notes:
- Images are identified by a server-generated UUID.
- Storage and index are in-memory; restart clears the registry.
- Uploaded image bytes are stored under a `tempfile` directory created at startup and cleaned up on shutdown.
- Image search requires CUDA. If you don't have CUDA, set `IMAGE_SEARCH_ENABLED=0`.

Environment variables:
- `IMAGE_SEARCH_ENABLED` (default: `1`) - set `0` to disable image search init
- `IMAGE_SEARCH_MAX_BYTES` (default: `20971520`) - max upload size per image

Endpoints (all under `/v1`):
- `POST /images` (multipart) - upload an image, returns `{id, ...}`
- `DELETE /images/{id}` - delete by UUID
- `GET /images` - list all images (UUID + original filename + metadata)
- `POST /images/search` - text search, returns `[{id, score}]`
- `GET /images/{id}/file` - download image bytes

Example (upload → list → search → download → delete):

```bash
# Upload (capture UUID)
ID=$(curl -s http://localhost:8000/v1/images \
  -F "file=@test/image01.jpg" | python -c "import sys, json; print(json.load(sys.stdin)['id'])")

# Upload multiple sample images
for f in test/image0{1..5}.jpg; do
  echo "Uploading $f"
  curl -s http://localhost:8000/v1/images -F "file=@$f"
  echo
done

# List
curl -s http://localhost:8000/v1/images

# Search
curl -s http://localhost:8000/v1/images/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"고양이", "limit": 5}'

# Download
curl -s http://localhost:8000/v1/images/$ID/file --output downloaded.bin

# Delete
curl -s -X DELETE http://localhost:8000/v1/images/$ID
```

## Voice generation (optional)

Enable voice model loading at startup:

```bash
export VOICE_ENABLED=1
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
