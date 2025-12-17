FROM python:3.12-slim

# -------------------------
# Environment
# -------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy backend code
# -------------------------
COPY . .

# -------------------------
# ⬇️ DOWNLOAD MODELS AT BUILD TIME
# -------------------------
RUN python - <<EOF
from download_models import download
download()
EOF

# -------------------------
# Sanity check (optional)
# -------------------------
RUN python - <<EOF
import os
print("Model files:", os.listdir("model"))
EOF

# -------------------------
# Expose port
# -------------------------
EXPOSE 8000

# -------------------------
# Run server
# -------------------------
CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT}
