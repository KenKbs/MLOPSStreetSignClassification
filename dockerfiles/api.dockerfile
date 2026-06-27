FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ARG API_MODEL_NAME=YOLO_eps420_bs8_lr0.005_fr10_x.pt

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

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN mkdir -p API_uploads/input API_uploads/output API_uploads/terminal_requests models
COPY models/${API_MODEL_NAME} models/${API_MODEL_NAME}
ENV MODEL_NAME=${API_MODEL_NAME}

ENTRYPOINT ["uv", "run", "uvicorn", "street_sign_project.main:app", "--host", "0.0.0.0", "--port", "8000"]
