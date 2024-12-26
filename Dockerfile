FROM python:3.10-slim

# 필요한 시스템 패키지 및 한글 폰트 설치
RUN apt-get update && apt-get install -y \
    fonts-nanum \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# 폰트 캐시 업데이트 - 이는 설치된 폰트를 시스템이 인식하도록 합니다
RUN fc-cache -f -v

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 기존 실행 명령어를 유지합니다
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]