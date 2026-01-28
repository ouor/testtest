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

image generation (stores output in R2 and returns the R2 key only):
```bash
curl -s http://localhost:8000/v1/r2/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A serene landscape with mountains and a river during sunset.","seed":42}'

// {"key":"images/generated/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.png"}
```

### Notes
- Model loads once at startup (FastAPI lifespan).
- Inference is serialized with a semaphore (`max_concurrency=1`).

Environment variables:
- `IMAGE_ENABLED` (default: `1`) - set `0` to skip image model loading

## Image search (upload + semantic search)

This feature lets you upload images, keep them in a server-side directory, and search by text using a CLIP-style embedding model.

Notes:
- Images are identified by a server-generated UUID.
- The vector index + image metadata are stored in `IMAGE_SEARCH_DB_PATH` (default: `app/image_search.db`).
- Uploaded image bytes are stored under `IMAGE_SEARCH_FILES_DIR` (default: `app/image_search_files/`).
- Restart keeps the registry, since both are persisted under `app/`.
- Image search requires CUDA. If you don't have CUDA, set `IMAGE_SEARCH_ENABLED=0`.

Embedding model:
- Hardcoded: `Bingsu/clip-vit-large-patch14-ko` (no `IMAGE_SEARCH_MODEL_NAME` env)

Environment variables:
- `IMAGE_SEARCH_ENABLED` (default: `1`) - set `0` to disable image search init
- `IMAGE_SEARCH_DB_PATH` (default: `app/image_search.db`) - SQLite file path for the vector DB (relative paths are resolved from the project root)
- `IMAGE_SEARCH_FILES_DIR` (default: `app/image_search_files/`) - directory where uploaded images are stored (relative paths are resolved from the project root)
- `IMAGE_SEARCH_MAX_ELEMENTS` (default: `50000`) - HNSW capacity for the vectorlite index
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

R2 voice generation (ref_audio provided as an R2 key; returns output R2 key only):

```bash
curl -s http://localhost:8000/v1/r2/voice/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "ref_audio_key": "voice/refs/ref.mp3",
    "ref_text": "아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든.. 요! 근데 그 쪽에서 자꾸 안 믿어주니까!",
    "text": "오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?",
    "language": "Korean"
  }'
```

Notes:
- Endpoint returns `audio/mpeg` bytes (mp3).
- MP3 encoding uses `torchaudio`; depending on your environment, an ffmpeg-backed build may be required.

## Cloudflare R2 (optional)

The server includes a small S3-compatible utility for Cloudflare R2 in [testtest/app/core/storage/r2.py](testtest/app/core/storage/r2.py).

Environment variables:
- `R2_ENABLED` (default: `0`) - set `1` to enable R2 client initialization
- `R2_ACCOUNT_ID` (required unless `R2_ENDPOINT_URL` is provided)
- `R2_ENDPOINT_URL` (optional) - override endpoint URL (otherwise derived from `R2_ACCOUNT_ID`)
- `R2_ACCESS_KEY_ID` (required)
- `R2_SECRET_ACCESS_KEY` (required)
- `R2_BUCKET_NAME` (required)
