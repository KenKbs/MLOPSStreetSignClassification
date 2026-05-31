import os
from datetime import datetime
from shlex import quote

from invoke import Context, task
from src.street_sign_project.data import create_yaml_dataset

WINDOWS = os.name == "nt"
PROJECT_NAME = "street_sign_project"
PYTHON_VERSION = "3.12"


def _hydra_override_args(overrides: dict[str, object]) -> str:
    """Build Hydra CLI override arguments from non-empty values."""
    return " ".join(f"{key}={quote(str(value))}" for key, value in overrides.items() if value is not None)


def _train_overrides(
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
) -> dict[str, object]:
    """Map train task arguments to Hydra override keys."""
    return {
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
    override_args = _hydra_override_args(
        _train_overrides(
            data_yaml=data_yaml,
            model_path=model_path,
            yolo_model_size=yolo_model_size,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            optimizer=optimizer,
            lr0=lr0,
            freeze=freeze,
            device=device,
            workers=workers,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_dir=wandb_dir,
        )
    )
    command = f"uv run src/{PROJECT_NAME}/train.py"
    if override_args:
        command = f"{command} {override_args}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def profile_train(
    ctx: Context,
    output_path: str | None = None,
    data_yaml: str | None = None,
    model_path: str | None = None,
    yolo_model_size: str | None = None,
    epochs: int | None = 1,
    batch_size: int | None = None,
    seed: int | None = None,
    optimizer: str | None = None,
    lr0: float | None = None,
    freeze: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = "disabled",
    wandb_dir: str | None = None,
) -> None:
    """Profile one training run with cProfile."""
    ctx.run("mkdir -p reports/profiling", echo=True, pty=not WINDOWS)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"reports/profiling/train-{timestamp}.prof"
    override_args = _hydra_override_args(
        _train_overrides(
            data_yaml=data_yaml,
            model_path=model_path,
            yolo_model_size=yolo_model_size,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            optimizer=optimizer,
            lr0=lr0,
            freeze=freeze,
            device=device,
            workers=workers,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_dir=wandb_dir,
        )
    )
    command = f"uv run python -m cProfile -o {quote(output_path)} src/{PROJECT_NAME}/train.py"
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
