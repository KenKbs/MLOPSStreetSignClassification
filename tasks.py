import os
import re
from datetime import datetime
from shlex import quote

from hydra import compose, initialize
from invoke.context import Context
from invoke.tasks import task
from loguru import logger
from street_sign_project.model import YOLOv26
from street_sign_project.train import train_model
from street_sign_project.utils import project_root

WINDOWS = os.name == "nt"
PROJECT_NAME = "street_sign_project"
PYTHON_VERSION = "3.12"


def _hydra_override_args(overrides: dict[str, object]) -> str:
    """Build Hydra CLI override arguments from non-empty values."""
    return " ".join(f"{key}={quote(str(value))}" for key, value in overrides.items() if value is not None)


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
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py create-yaml-dataset", echo=True, pty=not WINDOWS)


@task
def train(
    ctx: Context,
    data_yaml: str | None = None,
    model_path: str | None = None,
    base_model_name: str | None = None,
    yolo_model_size: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    optimizer: str | None = None,
    lr0: float | None = None,
    freeze: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    patience: int | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = None,
    wandb_dir: str | None = None,
    auto_save: bool = True,
    auto_cleanup: bool = False,
) -> None:
    """
    Train model.

    Parameters not passed here use the defaults from configs/config.yaml.
    """
    overrides = {
        "paths.data_yaml": data_yaml,
        "paths.model_path": model_path,
        "model.base_model_name": base_model_name,
        "model.yolo_model_size": yolo_model_size,
        "training.epochs": epochs,
        "training.batch_size": batch_size,
        "training.seed": seed,
        "training.optimizer": optimizer,
        "training.lr0": lr0,
        "training.freeze": freeze,
        "training.device": device,
        "training.workers": workers,
        "training.patience": patience,
        "wandb.entity": wandb_entity,
        "wandb.project": wandb_project,
        "wandb.mode": wandb_mode,
        "wandb.dir": wandb_dir,
    }
    override_args = [f"{key}={str(value)}" for key, value in overrides.items() if value is not None]
    # command = f"uv run src/{PROJECT_NAME}/train.py"
    # if override_args:
    # command = f"{command} {override_args}"
    # ctx.run(command, echo=True, pty=not WINDOWS)
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config", overrides=override_args)
    trained_model: YOLOv26 = train_model.__wrapped__(
        cfg
    )  # __wrapped__ da es sonst konflikte mit hydra.main in train.py gibt
    cleanup_indicator = True
    if auto_cleanup:
        if auto_save:
            cleanup_indicator = trained_model.cleanup_savings()
        else:
            logger.error("auto saving is turned off, therefore cleanup will NOT be executed")
    if auto_save and cleanup_indicator:
        trained_model.save_model(project_root() / "models" / f"{trained_model.name_string}.pt")


@task
def profile_train(
    ctx: Context,
    output_path: str | None = None,
    data_yaml: str | None = None,
    model_path: str | None = None,
    base_model_name: str | None = None,
    yolo_model_size: str | None = None,
    epochs: int | None = 1,
    batch_size: int | None = None,
    seed: int | None = None,
    optimizer: str | None = None,
    lr0: float | None = None,
    freeze: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    patience: int | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_mode: str | None = "disabled",
    wandb_dir: str | None = None,
) -> None:
    """
    Profile one training run with cProfile.
    WATCH OUT: Model is not saved
    """
    ctx.run("mkdir -p reports/profiling", echo=True, pty=not WINDOWS)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"reports/profiling/train-{timestamp}.prof"
    override_args = _hydra_override_args(
        {
            "paths.data_yaml": data_yaml,
            "paths.model_path": model_path,
            "model.base_model_name": base_model_name,
            "model.yolo_model_size": yolo_model_size,
            "training.epochs": epochs,
            "training.batch_size": batch_size,
            "training.seed": seed,
            "training.optimizer": optimizer,
            "training.lr0": lr0,
            "training.freeze": freeze,
            "training.device": device,
            "training.workers": workers,
            "training.patience": patience,
            "wandb.entity": wandb_entity,
            "wandb.project": wandb_project,
            "wandb.mode": wandb_mode,
            "wandb.dir": wandb_dir,
        }
    )
    command = f"uv run python -m cProfile -o {quote(output_path)} src/{PROJECT_NAME}/train.py"
    if override_args:
        command = f"{command} {override_args}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run pytest tests/", echo=True, pty=not WINDOWS)


@task
def test_coverage(ctx: Context) -> None:
    """Run tests with coverage reporting."""
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


@task
def tune(ctx: Context) -> None:
    """tuning the model. See configs/sweep.yaml for changing the params"""
    output = ctx.run("uv run wandb sweep configs/sweep.yaml", echo=True, pty=not WINDOWS)
    if output is None:
        raise ValueError("Output of wandb sweep was none. Check sweep.yaml for errors")
    full_output = output.stdout + output.stderr
    match = re.search(
        r"(?i)with\s+ID:(?:[^\w]*\d+m)?\s*([a-z0-9]{8})", full_output
    )  # complicated regex because terminal output is in yellow
    if match is None:
        raise ValueError("no ID found for the sweep check the output of >>>uv run wandb sweep configs/sweep.yaml<<<")
    id_string = match.group(1).strip()
    ctx.run(
        f"uv run wandb agent k-kubsch-ludwig-maximilian-university-of-munich/StreetSignClassification/{id_string}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def create_models_quality_yaml(ctx: Context) -> None:
    """creating a yaml for saving of 3 best models"""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py models-quality-yaml", echo=True, pty=not WINDOWS)


@task
def plot_images(ctx: Context) -> None:
    """plotting example images"""
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py plot-image-pred", echo=True, pty=not WINDOWS)
