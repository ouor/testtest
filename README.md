# AI Model Server (FastAPI)

이미지 생성, 이미지 시맨틱 검색, 음성 생성을 제공하는 FastAPI 기반 AI 모델 서버입니다.  
이미지 검색 및 일부 생성 결과는 **Cloudflare R2를 필수 스토리지**로 사용합니다.

모든 기능은 환경 변수로 독립적으로 활성화/비활성화할 수 있습니다.

---

## 1. 제공 기능 요약

- 🖼️ **이미지 생성**
  - 파일 바이너리 응답
  - R2 저장 후 key 반환
- 🔊 **참조 음성 기반 음성 생성**
  - 로컬 mp3 반환
  - R2 저장 후 key 반환
- 🔍 **이미지 업로드 + 텍스트 기반 시맨틱 검색**
  - 이미지 원본은 R2에 저장
  - CLIP 임베딩 기반 검색
- ☁️ **Cloudflare R2 연동 (필수/선택 혼합)**

---

## 2. 아키텍처 개요

- FastAPI 기반 REST API
- 모델은 서버 시작 시 1회 로드 (lifespan)
- 모든 추론은 세마포어로 직렬 처리 (`max_concurrency = 1`)
- 기능별 모듈화:
  - Image Generation
  - Voice Generation
  - Image Search (R2 기반)
  - R2 Storage Utility

---

## 3. Quick Start (이미지 생성만 사용)

### 3.1 실행

```bash
conda create -n default python=3.10
conda activate default
conda install forge:uv
uv pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
````

### 3.2 헬스 체크

`GET /healthz`: 서버 상태 확인  
`GET /readyz`: 모델 로드 완료 확인

```bash
curl http://localhost:8000/healthz | python -m json.tool
curl http://localhost:8000/readyz | python -m json.tool
```

---

## 4. 공통 설정 (Environment Variables)

- `IMAGE_SEARCH_DB_R2_KEY` (default: `image_search.db`) - R2 key used for DB restore/backup snapshots
- `IMAGE_SEARCH_DB_BACKUP_ENABLED` (default: `1`) - set `0` to disable DB backup/restore via R2
- `IMAGE_SEARCH_DB_BACKUP_INTERVAL_SECONDS` (default: `1800`) - periodic DB backup interval

### 기능 토글

| 기능     | 변수                   | 기본값 |
| ------ | -------------------- | --- |
| 이미지 생성 | IMAGE_ENABLED        | 1   |
| 이미지 검색 | IMAGE_SEARCH_ENABLED | 0   |
| 음성 생성  | VOICE_ENABLED        | 0   |
| R2 연동  | R2_ENABLED           | 0   |

---

## 5. 이미지 생성

### 5.1 로컬 이미지 생성

**Endpoint**

```
POST /v1/images/generate
```

**Example**

```bash
curl -s http://localhost:8000/v1/images/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A serene landscape with mountains and a river during sunset.",
    "seed": 42
  }' \
  --output out.png
```

---

### 5.2 R2 이미지 생성

**Endpoint**

```
POST /v1/r2/images/generate
```

**Example**

```bash
curl -s http://localhost:8000/v1/r2/images/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A serene landscape with mountains and a river during sunset.",
    "seed": 42
  }' | python -m json.tool
```

**Response**

```json
{
  "key": "images/generated/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.png"
}
```

---

### 참고 사항

* `IMAGE_ENABLED=0` 설정 시 이미지 모델을 로드하지 않습니다.
* 모델은 서버 시작 시 1회 로드됩니다.

---

## 6. 이미지 검색

> ⚠️ 이미지 검색 기능은 **Cloudflare R2**가 반드시 필요합니다.

### 6.1 개념 및 저장 구조

* 이미지 업로드 시:

  * 원본 바이트 → **R2 저장**
  * 서버는 UUID + `r2_key`만 관리
* CLIP 임베딩 및 벡터 인덱스:

  * SQLite (`IMAGE_SEARCH_DB_PATH`)에 영구 저장
* `IMAGE_SEARCH_FILES_DIR`:

  * 임베딩 처리 중 임시 파일
  * 레거시 로컬 이미지 레코드 용도

---

### 6.2 환경 변수

| 변수                        | 기본값                     | 설명         |
| ------------------------- | ----------------------- | ---------- |
| IMAGE_SEARCH_ENABLED      | 1                       | 이미지 검색 활성화 |
| IMAGE_SEARCH_DB_PATH      | app/image_search.db     | 벡터 DB      |
| IMAGE_SEARCH_FILES_DIR    | app/image_search_files/ | 임시/레거시 파일  |
| IMAGE_SEARCH_MAX_ELEMENTS | 50000                   | HNSW 용량    |
| IMAGE_SEARCH_MAX_BYTES    | 20971520                | 최대 업로드 크기  |

---

### 6.3 이미지 업로드

**Endpoint**

```
POST /v1/images
```

**Example**

```bash
curl -s http://localhost:8000/v1/images \
  -F "file=@test/image01.jpg" | python -m json.tool
```

---

### 6.4 이미지 목록 조회

**Endpoint**

```
GET /v1/images
```

**Example**

```bash
curl -s http://localhost:8000/v1/images | python -m json.tool
```

---

### 6.5 이미지 검색

**Endpoint**

```
POST /v1/images/search
```

**Example**

```bash
curl -s http://localhost:8000/v1/images/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "고양이",
    "limit": 5
  }' | python -m json.tool
```

---

### 6.6 이미지 다운로드 (R2 presigned URL redirect)

**Endpoint**

```
GET /v1/images/{id}/file
```

**Example (redirect 확인)**

```bash
curl -s -D - http://localhost:8000/v1/images/$ID/file
```

**Example (실제 다운로드)**

```bash
curl -L -s \
  http://localhost:8000/v1/images/$ID/file \
  --output out.jpg
```

---

### 6.7 이미지 삭제

**Endpoint**

```
DELETE /v1/images/{id}
```

**Example**

```bash
curl -s -X DELETE http://localhost:8000/v1/images/{id}
```

---

## 7. 음성 생성 (선택 기능)

### 7.1 활성화

```bash
export VOICE_ENABLED=1
```

---

### 7.2 로컬 음성 생성

**Endpoint**

```
POST /v1/voice/generate
```

**Example**

```bash
curl -s http://localhost:8000/v1/voice/generate \
  -F "ref_audio=@test/ref.mp3" \
  -F "ref_text=아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든.." \
  -F "text=오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?" \
  -F "language=Korean" \
  --output out.mp3
```

---

### 7.3 R2 음성 생성

**Endpoint**

```
POST /v1/r2/voice/generate
```

**Example**

```bash
curl -s http://localhost:8000/v1/r2/voice/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "ref_audio_key": "voice/refs/ref.mp3",
    "ref_text": "아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든..",
    "text": "오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?",
    "language": "Korean"
  }' | python -m json.tool
```

**Response**

```json
{
  "key": "voice/generated/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.mp3"
}
```

### 참고 사항

* 응답 포맷: `audio/mpeg`
* MP3 인코딩은 `torchaudio` 사용
* 환경에 따라 ffmpeg 지원이 필요할 수 있습니다.

---

## 8. Cloudflare R2 설정

### 개요

* S3-compatible API 사용
* 이미지 검색 기능에서는 **필수**
* 이미지/음성 생성에서는 **선택**

### 환경 변수

```bash
export R2_ENABLED=1
export R2_ACCOUNT_ID=xxxx
export R2_ENDPOINT_URL=https://<custom-endpoint>   # optional
export R2_ACCESS_KEY_ID=xxxx
export R2_SECRET_ACCESS_KEY=xxxx
export R2_BUCKET_NAME=xxxx
```

### 구현 위치

```
* `app/core/storage/r2.py`
```