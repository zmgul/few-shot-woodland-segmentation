# Official PyTorch image compatible with HPC CUDA 11.8
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Working directory
WORKDIR /app

# System dependencies & Cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Library installation
RUN pip install --no-cache-dir 'numpy>=1.24,<2'
RUN pip install --no-cache-dir poetry

# Copy only dependency files
COPY pyproject.toml ./

# Clean cache after installation
RUN poetry config virtualenvs.create false \
    && poetry lock \
    && poetry install --no-interaction --no-ansi --no-root \
    && rm -rf /root/.cache

# MLflow environment variable
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
