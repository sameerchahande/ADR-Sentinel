# ADR-Sentinel v2 - Docker image (python:3.11-slim + uvicorn on port 8000).
# The build step trains the model once and bakes the artefacts into the image so
# the container starts instantly and can serve the demo without any network access.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# curl is only needed for the container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# SIDER reference data + application code
COPY drug_names.tsv drug_atc.tsv meddra_all_se_tsv.gz ./
COPY sentinel ./sentinel
COPY static ./static
COPY adr_prediction_model.py main.py ./

# Train, evaluate, render the figures and cache the demo case at build time
RUN python -m sentinel.build_model

RUN useradd --create-home --shell /usr/sbin/nologin sentinel \
    && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "sentinel.api:app", "--host", "0.0.0.0", "--port", "8000"]