FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

EXPOSE 8000

# ⚠️ IMPORTANT: use $PORT for Render
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port $PORT"]
