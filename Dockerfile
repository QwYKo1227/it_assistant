FROM python:3.12-slim

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}

WORKDIR /app

RUN pip install --no-cache-dir uv -i "${PIP_INDEX_URL}"

COPY pyproject.toml uv.lock ./
COPY app ./app
COPY prompts ./prompts
COPY .env.example ./.env.example

RUN mkdir -p /app/output /app/logs \
    && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8001

CMD ["uvicorn", "app.query_process.api.query_service:app", "--host", "0.0.0.0", "--port", "8001"]
