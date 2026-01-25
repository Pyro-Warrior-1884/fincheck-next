FROM python:3.12-slim

# --------------------------------------------------
# Environment
# --------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# --------------------------------------------------
# System dependencies (OCR + OpenCV + Torch runtime)
# --------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python dependencies (cached layer)
# --------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt && \
    pip install --no-cache-dir pytesseract

# --------------------------------------------------
# Copy application code
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# Download models at build time
# --------------------------------------------------
RUN python backend/download_models.py

# --------------------------------------------------
# Sanity check (fail fast)
# --------------------------------------------------
RUN python - <<EOF
import pytesseract, os
print("✅ Tesseract:", pytesseract.get_tesseract_version())
models = os.listdir("backend/model")
assert models, "❌ No models found"
print("✅ Models:", models)
EOF

# --------------------------------------------------
# Create non-root user (Render-safe)
# --------------------------------------------------
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# --------------------------------------------------
# Run server (Render REQUIRED format)
# --------------------------------------------------
CMD ["sh", "-c", \
     "uvicorn backend.server:app \
      --host 0.0.0.0 \
      --port ${PORT:-8000} \
      --workers 2 \
      --timeout-keep-alive 30"]
