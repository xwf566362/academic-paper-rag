FROM python:3.11-slim

WORKDIR /app

# System dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models during build (cached in a layer)
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py && rm -rf /root/.cache/huggingface/hub/*.lock

# Copy application code
COPY . .

# Pre-download config.local.yaml if not present
RUN if [ ! -f config.local.yaml ]; then echo "" > config.local.yaml; fi

# Ports: FastAPI + Gradio
EXPOSE 8000 7860

# Entrypoint
CMD ["bash", "docker/start.sh"]
