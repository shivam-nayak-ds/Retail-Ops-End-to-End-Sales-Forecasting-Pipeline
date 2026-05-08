# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for stability
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/src"
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for LightGBM/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
# CRITICAL FIX: Force CPU-only PyTorch (skips 3GB+ of NVIDIA/CUDA downloads)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Install the project as a package (CRITICAL)
RUN pip install -e .

# Expose the API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
