# ============================================================
# DOCKERFILE — Chest X-Ray Diagnosis API
# ============================================================
# Builds the FastAPI prediction server.
# For the Streamlit dashboard use Dockerfile.dashboard
# ============================================================

FROM python:3.10-slim

# System deps for OpenCV (headless) + PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/          ./src/
COPY serving/      ./serving/
COPY services/     ./services/
COPY xray_models/  ./xray_models/
COPY logs/         ./logs/

# Create runtime dirs
RUN mkdir -p logs xray_models data

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "serving.xray_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
