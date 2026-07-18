# Getting started

This page shows how to set up the environment and run the most common project commands. All
project commands are exposed as [`invoke`](https://www.pyinvoke.org/) tasks in `tasks.py`; run
`uv run invoke --list` to see every available task.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
git clone <repo-url>
cd MLOPSStreetSignClassification
# install uv first: https://docs.astral.sh/uv/
uv sync --dev --locked   # create the .venv from the locked versions
```

`uv sync --dev --locked` reproducibly recreates the virtual environment from `uv.lock`. Prefix any
command with `uv run` to run it inside that environment.

## Data

The raw and preprocessed data as well as the trained models are versioned with **DVC** and stored
in a Google Cloud Storage remote. Pull them with:

```bash
uv run dvc pull
```

To (re)build the preprocessed dataset and the YOLO dataset YAML from the raw data:

```bash
uv run invoke preprocess-data   # remap labels + deterministic train/valid/test split
uv run invoke create-yaml       # write data/dataset.yaml
```

## Training

Training is configured with **Hydra** (`configs/config.yaml`) and launched through an invoke task.
Any config value can be overridden on the command line:

```bash
uv run invoke train --epochs 100 --batch-size 8 --lr0 0.005 --freeze 10
```

Runs are logged to **Weights & Biases** (metrics, per-epoch loss curves and the trained model as an
artifact). A Bayesian hyperparameter sweep is defined in `configs/sweep.yaml` and started with
`uv run invoke tune`.

## Testing and code quality

```bash
uv run invoke test            # run the unit and API tests (excludes performance tests)
uv run invoke test-coverage   # run the tests with coverage reporting
uv run ruff check . --fix     # lint (and auto-fix)
uv run ruff format .          # format
uv run pre-commit run --all-files
```

## Profiling

```bash
uv run invoke profile-train   # profile one training run with cProfile + snakeviz
```

## Serving the model

### FastAPI

```bash
uv run invoke start-local-api   # serve on http://localhost:8000 (docs at /docs)
```

The API exposes `POST /image_input/` (upload an image, receive an annotated image) and
`GET /monitoring/` (an Evidently data-drift report).

### Streamlit frontend

```bash
uv run invoke start-local-frontend   # starts the API and the Streamlit UI
```

### BentoML service

```bash
uv run invoke start-bento   # build, containerize and run the BentoML API locally
uv run invoke test-bento    # send one smoke-test image to the running service
```

### Load testing

```bash
uv run invoke stress-api          # headless Locust run
uv run invoke stress-api --ui     # Locust web UI
```

## Docker

```bash
uv run invoke docker-build      # build the train, evaluate and api images
uv run invoke docker-train      # run one training run in a container
uv run invoke docker-api        # serve the API in a container
```

## Deployment (Google Cloud Run)

```bash
uv run invoke deploy-api        # build, push and deploy the FastAPI service
uv run invoke deploy-bento      # deploy the BentoML service
uv run invoke deploy-frontend   # deploy the Streamlit frontend
```

The equivalent shell scripts live in `scripts/` and accept overrides via environment variables
(`PROJECT_ID`, `REGION`, `SERVICE_NAME`, `MODEL_NAME`, …).

## Building this documentation

```bash
uv run invoke build-docs   # build the static site
uv run invoke serve-docs   # serve the docs locally with live reload
```
