import os
from shlex import quote

from invoke import Context, task
from src.street_sign_project.data import create_yaml_dataset

WINDOWS = os.name == "nt"
PROJECT_NAME = "street_sign_project"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py export-mapping-to-csv", echo=True, pty=not WINDOWS)
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py preprocess", echo=True, pty=not WINDOWS)
    # TODO maybe, make paths editable in this task


@task
def create_yaml(ctx: Context) -> None:
    """
    Function to create a file to pass data to YOLO
    located in data.py
    """
    create_yaml_dataset()


@task
def train(
    ctx: Context,
    data_yaml: str | None = None,
    model_path: str | None = None,
    yolo_model_size: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    optimizer: str | None = None,
    lr0: float | None = None,
    freeze: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = None,
    wandb_dir: str | None = None,
) -> None:
    """
    Train model.

    Parameters not passed here use the defaults from configs/config.yaml.
    """
    # Define override values
    overrides = {
        "paths.data_yaml": data_yaml,
        "paths.model_path": model_path,
        "model.yolo_model_size": yolo_model_size,
        "training.epochs": epochs,
        "training.batch_size": batch_size,
        "training.seed": seed,
        "training.optimizer": optimizer,
        "training.lr0": lr0,
        "training.freeze": freeze,
        "training.device": device,
        "training.workers": workers,
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.mode": wandb_mode,
        "wandb.dir": wandb_dir,
    }

    # Glue together overwritten values
    override_args = " ".join(f"{key}={quote(str(value))}" for key, value in overrides.items() if value is not None)

    # Create command
    command = f"uv run src/{PROJECT_NAME}/train.py"
    if override_args:
        command = f"{command} {override_args}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
