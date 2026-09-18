FROM python:3.11-slim

WORKDIR /app

# sentence-transformers only requires torch>=1.11.0 (verified against its
# actual package metadata) -- the default PyPI torch wheel bundles CUDA and
# is 750MB+; installing the CPU-only build from PyTorch's own index first
# satisfies that constraint at a fraction of the size. pip won't reinstall
# it in the next step since the requirement is already satisfied.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

# Binding 0.0.0.0, not 127.0.0.1 -- required for the port mapping in
# docker-compose.yml to actually reach the app from outside the container.
# Verified: tested this exact command locally, confirmed /health, /docs and
# /openapi.json all return 200 before this ever went into a container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
