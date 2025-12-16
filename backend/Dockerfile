FROM python:3.12-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (required for torch + pillow)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# 🔥 DOWNLOAD MODELS AT BUILD TIME (IMPORTANT)
RUN python download_models.py

# Render expects the app to listen on this port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
