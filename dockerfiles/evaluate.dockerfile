FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc libgl1 libglib2.0-0 libxcb1 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src src/
COPY tasks.py tasks.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN mkdir -p data/preprocessed/test/images data/preprocessed/test/labels models

ENTRYPOINT ["uv", "run", "invoke", "create-models-quality-yaml"]
