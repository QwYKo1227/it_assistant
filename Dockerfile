FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY app ./app
COPY prompts ./prompts
COPY .env.example ./.env.example

RUN mkdir -p /app/output /app/logs \
    && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8001

CMD ["uvicorn", "app.query_process.api.query_service:app", "--host", "0.0.0.0", "--port", "8001"]
