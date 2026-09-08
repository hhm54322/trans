FROM node:22-alpine AS web-builder

WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DATABASE_PATH=/app/data/app.db \
    APP_CORS_ORIGINS=http://localhost:8000

WORKDIR /app
COPY api/requirements.txt ./api/requirements.txt
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i \
          -e 's#deb.debian.org/debian-security#mirrors.aliyun.com/debian-security#g' \
          -e 's#deb.debian.org/debian#mirrors.aliyun.com/debian#g' \
          /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-noto-cjk \
        fonts-noto-core \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-tha \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    -r api/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*
COPY api/ ./api/
COPY --from=web-builder /build/web/dist ./web/dist
RUN mkdir -p /app/data

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
