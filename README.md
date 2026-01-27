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
- `IMAGE_MODEL_NAME` (default: `Tongyi-MAI/Z-Image-Turbo`)
- `IMAGE_DEVICE` (default: `cuda`)
