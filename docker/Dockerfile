# syntax=docker/dockerfile:1
# Multi-stage build: install deps in a builder, copy into a slim runtime image.
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
# Non-root user — never run a public service as root.
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY models/ ./models/
ENV PYTHONUNBUFFERED=1 \
    FRAUD_MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
USER appuser
# Container-level healthcheck hits the liveness endpoint.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
