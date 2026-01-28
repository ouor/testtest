import sqlite3
import vectorlite_py
import torch
import glob
from PIL import Image
from transformers import AutoModel, AutoProcessor

# 1. 상수 및 모델 설정
MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"
DB_FILE = "image_search.db"
VECTOR_DIM = 768 # CLIP-ViT-L/14 고정
MAX_DATA = 50000 # 5만 건 설정

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# 2. DB 연결 및 확장 로드
conn = sqlite3.connect(DB_FILE)
conn.enable_load_extension(True)
conn.load_extension(vectorlite_py.vectorlite_path())

# 3. 테이블 생성 (성공한 예제의 문법 적용)
conn.execute("DROP TABLE IF EXISTS v_images")
conn.execute(f"""
    CREATE VIRTUAL TABLE v_images USING vectorlite(
        embedding float32[{VECTOR_DIM}] cosine, 
        hnsw(max_elements={MAX_DATA})
    )
""")
conn.execute("CREATE TABLE IF NOT EXISTS image_paths (id INTEGER PRIMARY KEY, path TEXT)")
print("--- 테이블 생성 성공 ---")

# 4. 이미지 벡터 추출 함수
def get_image_vec(path):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()[0].tobytes()

# 5. 인덱싱 실행 (./*.jpg)
image_files = glob.glob("./test/*.jpg")
print(f"인덱싱 시작: {len(image_files)}개 파일")

for i, path in enumerate(image_files):
    try:
        vec = get_image_vec(path)
        conn.execute("INSERT INTO v_images(rowid, embedding) VALUES (?, ?)", (i, vec))
        conn.execute("INSERT INTO image_paths(id, path) VALUES (?, ?)", (i, path))
    except Exception as e:
        print(f"실패 ({path}): {e}")
conn.commit()

# 6. "피자" 검색 수행
query_text = "피자"
inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    query_vec = model.get_text_features(**inputs).cpu().numpy()[0].tobytes()

# 성공 예제에서 사용한 knn_search 쿼리 적용
search_sql = """
    SELECT p.path, v.distance 
    FROM v_images v 
    JOIN image_paths p ON v.rowid = p.id 
    WHERE knn_search(v.embedding, knn_param(?, 1))
"""
res = conn.execute(search_sql, (query_vec,)).fetchone()

if res:
    print(f"\n질문: {query_text}")
    print(f"매칭 이미지: {res[0]} (거리: {res[1]:.4f})")
else:
    print("결과를 찾을 수 없습니다.")

conn.close()