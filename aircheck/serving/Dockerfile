# CPU image for the MAMMAL prediction API (api.py).
# Build from the repo root:  docker build -f aircheck/serving/Dockerfile -t mammal-api .
# Run:  docker run -p 8090:8080 -v <models folder>:/models -e MODELS_CONFIG=/models/models.json mammal-api
# Models are not in the image; they are mounted at run time (on Cloud Run, from GCS).

FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# g++ builds a few wheels; libxrender1/libxext6 are needed by RDKit
RUN apt-get update && apt-get install -y --no-install-recommends g++ libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 1. CPU-only PyTorch (much smaller than the CUDA build), before the repo pulls in the CUDA one
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2. The MAMMAL code and its dependencies from pyproject.toml. Editable, so its data files are used in place.
#    This also brings TensorFlow, which fuse-med-ml requires; scoring never calls it.
WORKDIR /app
COPY pyproject.toml README.md LICENSE.txt ./
COPY mammal ./mammal
RUN pip install -e .

# 3. API packages, plus LightGBM and RDKit at the training env's versions, and the API code
COPY aircheck/serving/requirements-serving.txt ./serving/
RUN pip install -r serving/requirements-serving.txt lightgbm==4.7.0 rdkit==2026.9.1
COPY aircheck/serving/*.py ./serving/

WORKDIR /app/serving
ENV PORT=8080 MAX_SMILES=50 BATCH_SIZE=16 DEVICE=cpu
# The tokenizer and, for a pretrained-embedding model, the base model are downloaded from Hugging Face on the
# first request unless HF_HOME holds them; mount a cache there to avoid that
ENV HF_HOME=/cache/huggingface
EXPOSE 8080
# Cloud Run sets PORT; MODELS_CONFIG must be given at run time
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
