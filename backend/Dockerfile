FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
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
# Download ML models at build time
# -------------------------
RUN python download_models.py

# Fail build if models missing
RUN ls -lh /app/model

# -------------------------
# Expose & run
# -------------------------
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
