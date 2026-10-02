# syntax=docker/dockerfile:1
FROM python:3.12-slim

# เปลี่ยนแหล่งดาวน์โหลดแพ็กเกจได้ตอน build ถ้าเน็ตไป pypi.org ช้า
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    NUMBA_CACHE_DIR=/data/.numba_cache \
    HOME=/tmp \
    PYTEST_ADDOPTS="-p no:cacheprovider"

WORKDIR /app

COPY requirements.txt .
# cache mount: build ครั้งต่อไปไม่ต้องโหลดแพ็กเกจซ้ำ
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "$PIP_INDEX_URL" -r requirements.txt

COPY goldbtc ./goldbtc
COPY tests ./tests

EXPOSE 8000
ENTRYPOINT ["python", "-m", "goldbtc"]
CMD ["--help"]
