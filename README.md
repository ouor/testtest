# Common AI Server

---

## 1. 개요 (Overview)

FastAPI 기반의 멀티모달 AI 모델 서버입니다. 이미지 생성, CLIP 기반 시맨틱 검색, 음성 합성 기능을 제공하며, Cloudflare R2를 영구 저장소로 활용합니다. 모든 기능은 환경 변수를 통해 모듈식으로 제어됩니다.

## 2. 아키텍처 (Architecture)

* **동기화:** `max_concurrency=1` 세마포어를 통해 GPU 리소스를 보호하며 순차 추론을 수행합니다.
* **저장소:**
* **Local:** 임시 파일 처리 및 SQLite(벡터 인덱스) 저장.
* **Cloudflare R2:** 최종 결과물 저장, DB 백업 스냅샷 관리.



---

## 3. 기능 요약 (Features)

* **Image:** OpenAI GPT-4.1 기반 프롬프트 최적화 후 이미지 생성.
* **Search:** CLIP 임베딩을 이용한 텍스트-이미지 간 시맨틱 검색.
* **Voice:** 참조 오디오를 활용한 Zero-shot 보이스 클로닝 및 TTS.

---

## 4. Quick Start

```bash
conda create -n ai-server python=3.10 -y
conda activate ai-server
conda install forge:uv -y
uv pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

```

* **Health Check:** `GET /healthz` (서버 상태), `GET /readyz` (모델 로드 상태)

---

## 5. 전체 API 엔드포인트 및 예제

### 5.1 이미지 생성 (Image Generation)

OpenAI GPT를 거쳐 생성된 프롬프트로 이미지를 생성합니다.

* **로컬 생성** (`POST /v1/images/generate`)
```bash
curl -s http://localhost:8000/v1/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"title": "햄릿", "description": "복수와 광기의 비극"}' --output out.png

```


* **R2 저장형 생성** (`POST /v1/r2/images/generate`)
```bash
curl -s http://localhost:8000/v1/r2/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"title": "햄릿", "description": "복수와 광기의 비극"}' | jq

```



### 5.2 이미지 검색 (Image Search - R2 필수)

`project_id` 기반으로 이미지를 관리하며, 첫 업로드 시 프로젝트가 자동 생성됩니다.

* **이미지 업로드** (`POST /v1/projects/{project_id}/images`)
```bash
curl -F "file=@test.jpg" http://localhost:8000/v1/projects/my_project/images

```


* **목록 조회** (`GET /v1/projects/{project_id}/images`)
```bash
curl http://localhost:8000/v1/projects/my_project/images

```


* **시맨틱 검색** (`POST /v1/projects/{project_id}/images/search`)
```bash
curl -X POST http://localhost:8000/v1/projects/my_project/images/search \
  -H 'Content-Type: application/json' -d '{"query": "고양이", "limit": 5}'

```


* **파일 다운로드 (Redirect)** (`GET /v1/projects/{project_id}/images/{id}/file`)
```bash
curl -L http://localhost:8000/v1/projects/my_project/images/123/file --output down.jpg

```


* **이미지 삭제** (`DELETE /v1/projects/{project_id}/images/{id}`)
```bash
curl -X DELETE http://localhost:8000/v1/projects/my_project/images/123

```



### 5.3 음성 생성 (Voice Generation)

* **로컬 생성 (Multipart)** (`POST /v1/voice/generate`)
```bash
curl -F "ref_audio=@ref.mp3" -F "ref_text=안녕" -F "text=반가워" \
  http://localhost:8000/v1/voice/generate --output voice.mp3

```


* **R2 기반 생성 (JSON)** (`POST /v1/r2/voice/generate`)
*R2의 `{VOICE_REMOTE_PREFIX}{user_id}.mp3` 파일을 자동 참조합니다.*
```bash
curl -X POST http://localhost:8000/v1/r2/voice/generate \
  -H 'Content-Type: application/json' -d '{"user_id": "user1", "text": "반가워"}'

```



---

## 6. .env 환경변수 설정

### 6.1 기능 활성화 및 DB

```bash
IMAGE_ENABLED=1        # 이미지 생성 활성
IMAGE_SEARCH_ENABLED=1 # 이미지 검색 활성
VOICE_ENABLED=1        # 음성 생성 활성
R2_ENABLED=1           # R2 스토리지 활성

IMAGE_SEARCH_DB_PATH=app/image_search.db
IMAGE_SEARCH_DB_BACKUP_ENABLED=1
IMAGE_SEARCH_DB_BACKUP_INTERVAL_SECONDS=1800

```

### 6.2 외부 서비스 연동

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_PROMPT_MODEL=gpt-4.1

# Cloudflare R2
R2_ACCOUNT_ID=xxxx
R2_ACCESS_KEY_ID=xxxx
R2_SECRET_ACCESS_KEY=xxxx
R2_BUCKET_NAME=my-bucket

```

### 6.3 경로 및 설정값

```bash
IMAGE_REMOTE_PREFIX="AI/POSTER/"
IMAGE_SEARCH_REMOTE_PREFIX="AI/SEARCH/"
VOICE_REMOTE_PREFIX="AI/VOICE/"

IMAGE_SEARCH_MAX_ELEMENTS=50000
IMAGE_SEARCH_MAX_BYTES=20971520

```