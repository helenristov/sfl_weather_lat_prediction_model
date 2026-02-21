# ── SFL Scientific Take-Home Challenge ───────────────────────────────────────
# Python 3.10 slim image with all required data-science dependencies.
FROM python:3.10-slim

LABEL maintainer="candidate"
LABEL description="Weather station latitude prediction pipeline"

# System dependencies for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY sfl_latitude_prediction.ipynb .
COPY run_pipeline.py .

# Copy data directories (expected structure: data/PS1/ and data/PS2/)
COPY data/ ./data/

# Default command: run the Python pipeline (not Jupyter, for Docker automation)
CMD ["python", "run_pipeline.py"]
