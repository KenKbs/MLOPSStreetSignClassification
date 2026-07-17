> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
  * To run tests with coverage reporting, use `uv run invoke test-coverage`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, use `uv run invoke --list` or refer to the
    `tasks.py` file.
  * To run the API Docker container locally, use `uv run invoke docker-api`.
  * To start the Streamlit frontend locally, use `uv run invoke start-local-frontend`.
  * To deploy the Streamlit frontend to Cloud Run, use `uv run invoke deploy-frontend` or `./scripts/deploy_frontend_cloudrun.sh`.
  * To build, containerize, and run the BentoML API Docker image locally, use `uv run invoke start-bento`.
  * To send one smoke-test request to the running BentoML API, use `uv run invoke test-bento`.
  * To run the default headless API stress test, use `uv run invoke stress-test`.
  * To start the Locust UI for API stress testing, use `uv run invoke stress-test --ui`.
* The primary API deployment runs through `.github/workflows/deploy_api.yaml` on relevant pushes to `main` or by
  manually starting the workflow. It tests the project, pulls the configured model from DVC, builds the API image with
  Cloud Build, and deploys it to Cloud Run.
  * Configure the `GCP_API_DEPLOY_CREDENTIALS` GitHub Actions secret and the repository variables documented in the
    workflow.
* To build, push, and deploy the API container to Cloud Run locally as a fallback, use
  `./scripts/deploy_api_cloudrun.sh`.
  * Override defaults with environment variables such as `PROJECT_ID`, `REGION`, `REPOSITORY`, `SERVICE_NAME`,
    `MODEL_NAME`, `TAG`, `CPU`, `MEMORY`, and `ALLOW_UNAUTHENTICATED`.
* To build, push, and deploy the Streamlit frontend container to Cloud Run, use `./scripts/deploy_frontend_cloudrun.sh`.
  * Override defaults with environment variables such as `PROJECT_ID`, `REGION`, `REPOSITORY`, `SERVICE_NAME`,
    `API_SERVICE_NAME`, `API_URL`, `TAG`, `CPU`, `MEMORY`, and `ALLOW_UNAUTHENTICATED`.
* To build, push, and deploy the BentoML API container to Cloud Run, use `uv run invoke deploy-bento` or
  `./scripts/deploy_bento_cloudrun.sh`.
  * Override defaults with environment variables such as `PROJECT_ID`, `REGION`, `REPOSITORY`, `SERVICE_NAME`,
    `IMAGE_NAME`, `MODEL_NAME`, `TAG`, `CPU`, `MEMORY`, `MIN_INSTANCES`, `MAX_INSTANCES`, and
    `ALLOW_UNAUTHENTICATED`.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project will be using `mkdocs` for documentation. To build the docs locally, use
    `uv run mkdocs serve`
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
