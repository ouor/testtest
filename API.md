# Common AI Server – API 문서

- Base URL: `http://localhost:8000`
- API prefix: `/v1`
- Content types:
  - JSON: `application/json`
  - Multipart: `multipart/form-data`
  - Image bytes: `image/png`
  - Audio bytes: `audio/mpeg`

## 공통: 에러 응답

서버는 예외 발생 시 아래 형태의 JSON을 반환합니다.

```json
{
  "error": {
    "code": "SOME_CODE",
    "message": "Human readable message",
    "detail": {"optional": true}
  }
}
```

- `code` 예시: `INVALID_PROJECT`, `INVALID_ID`, `PROJECT_NOT_FOUND`, `R2_NOT_ENABLED`, `MODEL_LOAD_FAILED`, `OUT_OF_MEMORY`, `INFERENCE_FAILED`

---

## Status

### GET `/healthz`

- 설명: 서버 프로세스가 살아있는지 확인

**Request**: 없음

**Response 200**

```json
{"status":"ok"}
```

**curl**

```bash
curl -s http://localhost:8000/healthz | python -m json.tool
```

---

### GET `/readyz`

- 설명: 모델 로딩 여부 기반 readiness

**Request**: 없음

**Response 200**

```json
{
  "server_ready": true,
  "zimage_turbo_ready": true,
  "qwen3_tts_ready": false,
  "image_search_ready": true
}
```

**curl**

```bash
curl -s http://localhost:8000/readyz | python -m json.tool
```

---

## Image Generation

### POST `/v1/images/generate`

- 설명: 연극 포스터 이미지를 생성합니다.
- 입력은 `title`/`description`만 받고, 서버 내부에서 LLM으로 프롬프트를 만든 뒤 이미지 생성합니다.
- 응답은 PNG 바이트입니다.

**Request (JSON)**

```json
{
  "title": "햄릿",
  "description": "덴마크 왕자의 복수와 광기를 다룬 비극. 어두운 궁정, 배신, 유령의 계시."
}
```

**Response 200**

- Content-Type: `image/png`
- Body: PNG bytes

**curl**

```bash
curl -s http://localhost:8000/v1/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"title":"햄릿","description":"덴마크 왕자의 복수와 광기를 다룬 비극."}' \
  --output out.png
```

---

### POST `/v1/r2/images/generate`

- 설명: 포스터를 생성한 뒤 R2에 업로드하고 key를 반환합니다.

**Request (JSON)**

```json
{
  "title": "햄릿",
  "description": "덴마크 왕자의 복수와 광기를 다룬 비극.",
  "key": "images/generated/custom.png"
}
```

- `key`는 선택(optional)입니다. 없으면 서버가 UUID 기반으로 생성합니다.

**Response 200 (JSON)**

```json
{"key":"AI/POSTER/images/generated/custom.png"}
```

**curl**

```bash
curl -s http://localhost:8000/v1/r2/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"title":"햄릿","description":"덴마크 왕자의 복수와 광기를 다룬 비극.","key":"images/generated/custom.png"}' | python -m json.tool
```

---

## Voice Generation

### POST `/v1/voice/generate`

- 설명: 레퍼런스 오디오(ref_audio) + 레퍼런스 텍스트(ref_text) 기반으로 음성을 생성합니다.
- 응답은 MP3 바이트입니다.

**Request (multipart/form-data)**

- `ref_audio`: 파일(필수)
- `ref_text`: 문자열(필수)
- `text`: 생성할 문장(필수)
- `language`: 언어(필수, 예: `Korean`)

**Response 200**

- Content-Type: `audio/mpeg`
- Body: MP3 bytes

**curl**

```bash
curl -s http://localhost:8000/v1/voice/generate \
  -F "ref_audio=@test/ref.mp3" \
  -F "ref_text=아이.. 그게 참.. 난 정말 진심으로 말하고 있는거거든.." \
  -F "text=오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?" \
  -F "language=Korean" \
  --output out.mp3
```

---

### POST `/v1/r2/voice/generate`

- 설명: R2에 저장된 레퍼런스 파일을 `user_id`로 찾아 음성을 생성한 뒤, 결과 MP3를 R2에 업로드하고 key를 반환합니다.

**레퍼런스 규칙**

- `VOICE_REMOTE_PREFIX`가 `AI/VOICE/`일 때:
  - 오디오: `AI/VOICE/{user_id}.mp3`
  - 텍스트: `AI/VOICE/{user_id}.txt`

**Request (JSON)**

```json
{
  "user_id": "user_001",
  "text": "오전 10시 30분에 예정된 미팅 일정을 다시 한번 확인해 주시겠어요?",
  "language": "Korean",
  "key": "voice/generated/user_001.mp3"
}
```

- `key`는 선택(optional)입니다. 없으면 서버가 UUID 기반으로 생성합니다.

**Response 200 (JSON)**

```json
{"key":"AI/VOICE/voice/generated/user_001.mp3"}
```

**curl**

```bash
curl -s http://localhost:8000/v1/r2/voice/generate \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_001","text":"안녕하세요.","language":"Korean","key":"voice/generated/user_001.mp3"}' | python -m json.tool
```

---

## Image Search

- 이미지 검색은 R2가 필요합니다.
- 프로젝트는 별도 생성 API가 없고, 해당 `project_id`로 첫 업로드 시 자동 생성됩니다.
- 업로드 전에 존재하지 않는 `project_id`로 `list/search/get/delete` 호출 시 `404 PROJECT_NOT_FOUND`가 반환됩니다.

### POST `/v1/projects/{project_id}/images`

- 설명: 이미지를 업로드하고, CLIP 임베딩을 생성해 프로젝트별 벡터DB에 저장합니다.

**Path params**

- `project_id`: `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`

**Request (multipart/form-data)**

- `file`: 이미지 파일(필수, `image/*`)

**Response 200 (JSON)**

```json
{
  "project_id": "default",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "r2_key": "AI/SEARCH/default/550e8400-e29b-41d4-a716-446655440000.jpg",
  "original_filename": "image01.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 123456
}
```

**curl**

```bash
PROJECT_ID=default
curl -s http://localhost:8000/v1/projects/$PROJECT_ID/images \
  -F "file=@test/image01.jpg" | python -m json.tool
```

---

### GET `/v1/projects/{project_id}/images`

- 설명: 프로젝트에 등록된 이미지 메타데이터 목록을 반환합니다.

**Response 200 (JSON)**

```json
{
  "images": [
    {
      "project_id": "default",
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "r2_key": "AI/SEARCH/default/550e8400-e29b-41d4-a716-446655440000.jpg",
      "original_filename": "image01.jpg",
      "content_type": "image/jpeg",
      "size_bytes": 123456
    }
  ]
}
```

**curl**

```bash
PROJECT_ID=default
curl -s http://localhost:8000/v1/projects/$PROJECT_ID/images | python -m json.tool
```

---

### POST `/v1/projects/{project_id}/images/search`

- 설명: 텍스트 쿼리를 임베딩 후 프로젝트 내부에서 유사 이미지들을 검색합니다.

**Request (JSON)**

```json
{
  "query": "고양이",
  "limit": 5
}
```

**Response 200 (JSON)**

```json
{
  "results": [
    {
      "project_id": "default",
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "r2_key": "AI/SEARCH/default/550e8400-e29b-41d4-a716-446655440000.jpg",
      "score": 0.8123,
      "original_filename": "image01.jpg",
      "content_type": "image/jpeg",
      "size_bytes": 123456
    }
  ]
}
```

**curl**

```bash
PROJECT_ID=default
curl -s http://localhost:8000/v1/projects/$PROJECT_ID/images/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"고양이","limit":5}' | python -m json.tool
```

---

### GET `/v1/projects/{project_id}/images/{image_id}/file`

- 설명: 이미지 다운로드를 위해 R2 presigned URL로 `307 Temporary Redirect`를 반환합니다.

**Response**

- `307` + `Location: https://...` 헤더

**curl (redirect 확인)**

```bash
PROJECT_ID=default
ID="550e8400-e29b-41d4-a716-446655440000"
curl -s -D - http://localhost:8000/v1/projects/$PROJECT_ID/images/$ID/file -o /dev/null
```

**curl (실제 다운로드)**

```bash
PROJECT_ID=default
ID="550e8400-e29b-41d4-a716-446655440000"
curl -L -s http://localhost:8000/v1/projects/$PROJECT_ID/images/$ID/file --output out.jpg
```

---

### DELETE `/v1/projects/{project_id}/images/{image_id}`

- 설명: 프로젝트에서 이미지를 삭제합니다 (DB + R2 오브젝트).

**Response 204**

- Body 없음

**curl**

```bash
PROJECT_ID=default
ID="550e8400-e29b-41d4-a716-446655440000"
curl -s -X DELETE http://localhost:8000/v1/projects/$PROJECT_ID/images/$ID -D - -o /dev/null
```
