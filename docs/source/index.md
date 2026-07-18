# street_sign_project

Documentation for **street_sign_project** — an end-to-end MLOps pipeline for **traffic-sign
detection and classification**, built for the DTU course *02476 Machine Learning Operations*.

## What the project does

The project fine-tunes a **YOLO (v26, Ultralytics)** object-detection model to detect and classify
traffic signs in images. Two public datasets — the German *GTSDB* and an Italian street-sign set —
are merged into a single canonical label space of **73 sign classes** (see
`configs/street_sign_class_mapping.csv`), split deterministically into train/validation/test, and
used to train a model that is then served through a web API and a browser frontend.

## MLOps stack

| Concern | Tooling |
| --- | --- |
| Environment & dependencies | `uv` (`pyproject.toml` + `uv.lock`) |
| Configuration | `Hydra` (`configs/config.yaml`), `wandb` sweeps (`configs/sweep.yaml`) |
| Data & model versioning | `DVC` with a Google Cloud Storage remote |
| Experiment tracking & registry | `Weights & Biases` (metrics, artifacts, staging → production) |
| Code quality | `ruff` (lint + format), `mypy`, `pre-commit` |
| Testing | `pytest` (unit, API, performance), `coverage`, `locust` |
| CI/CD | GitHub Actions (tests, lint, pre-commit, data checker, model staging, cloud build) |
| Containers | Docker (train / evaluate / API / frontend) + Cloud Build |
| Serving | `FastAPI`, a specialised `BentoML` service, and a `Streamlit` frontend on Cloud Run |
| Monitoring | `Evidently` data-drift reports over image features collected in production |

## Where to go next

- **[Getting started](getting_started.md)** — install the environment and run the most common
  commands (data, training, testing, API, frontend, docker, deployment).
- **API reference** — auto-generated documentation of the Python code, grouped into
  [Data pipeline](reference/data.md), [Model & training](reference/models.md),
  [Serving & APIs](reference/serving.md), [Monitoring](reference/monitoring.md) and
  [Utilities](reference/utilities.md).

The full project write-up (design decisions, cloud setup, results) lives in the exam report at
`reports/README.md`.
