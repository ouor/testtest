#!/bin/bash

# 1. 변수 설정
P_VERSION="1.4.7"
P_FILE="pgrok_${P_VERSION}_linux_amd64.tar.gz"
P_URL="https://github.com/pgrok/pgrok/releases/download/v${P_VERSION}/${P_FILE}"
INSTALL_PATH="/usr/local/bin/pgrok"

echo "--- pgrok 클라이언트 전역 설치를 시작합니다 ---"

# 2. 다운로드 및 압축 해제
echo "[1/4] pgrok 다운로드 중..."
wget -q --show-progress $P_URL

echo "[2/4] 압축 해제 중..."
tar -xzf $P_FILE

# 3. 공용 경로로 이동 (sudo 권한 필요)
echo "[3/4] /usr/local/bin으로 파일 이동 중... (비밀번호를 요청할 수 있습니다)"
sudo mv pgrok /usr/local/bin/
sudo chmod +x /usr/local/bin/pgrok

# 4. 정리
rm -f $P_FILE
echo "✅ [4/4] 설치 및 경로 설정 완료!"

echo "------------------------------------------------"
echo "🚀 이제 어디서든 아래 명령어를 바로 사용하세요:"
echo ""
echo "1️⃣  클라이언트 초기화 (서버 주소 및 토큰 설정):"
echo "    pgrok init --remote-addr example.com:2222 --forward-addr http://localhost:3000 --token {YOUR_TOKEN}"
echo ""
echo "2️⃣  HTTP 터널 개방 (예: 8000 포트):"
echo "    pgrok http 8000"
echo "------------------------------------------------"