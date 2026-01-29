# common-ai-server demo (Gradio)

This is a standalone Gradio UI that calls the FastAPI server over HTTP.
It intentionally does **not** import any server internals.

## Run

1) Start the API server (in another terminal):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2) Install demo deps (recommended to use a separate venv):

```bash
pip install -r demo/requirements.txt
```

3) Run Gradio:

```bash
python -m demo
```

## Env vars

- `DEMO_API_BASE_URL` (default: `http://localhost:8000`)
- `DEMO_TIMEOUT_SECONDS` (default: `60`)
