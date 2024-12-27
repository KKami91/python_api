FROM python:3.10-slim

# 필요한 시스템 패키지 및 한글 폰트 설치
RUN apt-get update && apt-get install -y \
    fonts-nanum \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# 폰트 캐시 업데이트
RUN fc-cache -f -v

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 폰트 파일 직접 복사 (옵션)
COPY fonts/ /usr/share/fonts/truetype/
RUN fc-cache -f -v

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]