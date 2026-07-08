import os
import pstats
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from shlex import quote
from urllib.parse import urlparse

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


# Helper functions
def _hydra_override_args(overrides: dict[str, object]) -> str:
    """Build Hydra CLI override arguments from non-empty values."""
    return " ".join(f"{key}={quote(str(value))}" for key, value in overrides.items() if value is not None)


def _default_profile_output_path() -> Path:
    """Build the default cProfile output path for a training run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("reports") / "profiling" / f"train-{timestamp}.prof"


def _print_profile_stats(profile_path: Path, limit: int) -> None:
    """Print terminal summaries for a cProfile output file."""
    for sort_by, title in (("cumulative", "Top cumulative time"), ("time", "Top self time")):
        print(f"\n{title} from {profile_path}")
        pstats.Stats(str(profile_path)).strip_dirs().sort_stats(sort_by).print_stats(limit)


def _is_port_open(host: str, port: int, timeout_seconds: float = 0.5) -> bool:
    """Check whether a TCP host:port is reachable."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_seconds)
        return sock.connect_ex((host, port)) == 0


def _start_local_api_detached(ctx: Context, port: int, model: str | None = None, reload: bool = False) -> None:
    """Start local FastAPI server as a detached process.
    This helper function is needed to be able to start the local API and the local frontend in the same command"""
    command = f"uv run uvicorn --port {port} street_sign_project.fast_api:app"
    if reload:
        command = f"{command} --reload --reload-dir src/{PROJECT_NAME}"
    if model is not None:
        command = f"MODEL_NAME={quote(model)} {command}"

    if WINDOWS:
        detached_command = f"start /B {command}"
    else:
        detached_command = f"nohup {command} >/tmp/street-sign-api-{port}.log 2>&1 &"

    ctx.run(detached_command, echo=True, pty=False)

    for _ in range(20):
        if _is_port_open("127.0.0.1", port):
            return
        time.sleep(0.5)

    logger.warning(f"API startup was triggered but port {port} is not reachable yet.")


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


@task(auto_shortflags=False)
def profile_train(
    ctx: Context,
    output_path: str | None = None,
    terminal_stats: bool = True,
    stats_limit: int = 30,
    snakeviz: bool = True,
    snakeviz_server: bool = False,
    snakeviz_host: str = "127.0.0.1",
    snakeviz_port: int = 8080,
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
    if output_path is None:
        profile_path = _default_profile_output_path()
    else:
        profile_path = Path(output_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
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
    # Long running comand, invoke is active until snakeViz server is stopped with CTRL+C
    command = f"uv run python -m cProfile -o {quote(str(profile_path))} src/{PROJECT_NAME}/train.py"
    if override_args:
        command = f"{command} {override_args}"
    ctx.run(command, echo=True, pty=not WINDOWS)
    if terminal_stats:
        _print_profile_stats(profile_path, stats_limit)
    if snakeviz:
        snakeviz_args = f"--hostname {quote(snakeviz_host)} --port {snakeviz_port}"
        if snakeviz_server:
            snakeviz_args = f"{snakeviz_args} --server"
        ctx.run(f"uv run snakeviz {snakeviz_args} {quote(str(profile_path))}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run pytest tests/ --ignore=tests/performancetests/", echo=True, pty=not WINDOWS)


@task
def test_coverage(ctx: Context) -> None:
    """Run tests with coverage reporting."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task(auto_shortflags=False, aliases=("stress-API",))
def stress_api(
    ctx: Context,
    host: str = "http://localhost:8000",
    users: int = 500,
    runtime: str = "2m",
    spawn_rate: float = 10.0,
    ui: bool = False,
) -> None:
    """Run the Locust API stress test."""
    locustfile = quote("tests/performancetests/locustfile.py")
    command = f"uv run locust -f {locustfile} --host {quote(host)}"
    if ui:
        ctx.run(command, echo=True, pty=not WINDOWS)
        return

    command = f"{command} --headless --users {users} --spawn-rate {spawn_rate} --run-time {quote(runtime)}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(f"docker compose --progress={progress} build train", echo=True, pty=not WINDOWS)
    ctx.run(f"docker compose --progress={progress} build evaluate", echo=True, pty=not WINDOWS)
    ctx.run(f"docker compose --progress={progress} build api", echo=True, pty=not WINDOWS)


@task
def docker_train(ctx: Context) -> None:
    """Run the training Docker container."""
    ctx.run("docker compose run --rm train", echo=True, pty=not WINDOWS)


@task
def docker_evaluate(ctx: Context) -> None:
    """Run the evaluation Docker container."""
    ctx.run("docker compose run --rm evaluate", echo=True, pty=not WINDOWS)


@task
def docker_api(ctx: Context) -> None:
    """Run the API Docker container."""
    ctx.run("docker compose run --rm --service-ports api", echo=True, pty=not WINDOWS)


@task(name="start-bento", auto_shortflags=False)
def start_bento(
    ctx: Context,
    bento: str = "street-sign-classifier:latest",
    image_tag: str = "street-sign-bento-api:latest",
    host_port: int = 3000,
    container_port: int = 3000,
    progress: str = "plain",
) -> None:
    """Build, containerize, and run the BentoML API Docker image locally."""
    ctx.run("uv run bentoml build", echo=True, pty=not WINDOWS)
    ctx.run(
        f"uv run bentoml containerize {quote(bento)} --image-tag {quote(image_tag)} --opt progress={quote(progress)}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(f"docker run --rm -p {host_port}:{container_port} {quote(image_tag)}", echo=True, pty=not WINDOWS)


@task(name="test-bento", auto_shortflags=False)
def test_bento(
    ctx: Context,
    host: str = "http://localhost:3000",
    image: str | None = None,
    output: str | None = None,
) -> None:
    """Send one smoke-test image to the running BentoML API service."""
    command = f"uv run python tests/apitests/bentoml_client.py --host {quote(host)}"
    if image is not None:
        command = f"{command} --image {quote(image)}"
    if output is not None:
        command = f"{command} --output {quote(output)}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task(auto_shortflags=False)
def deploy_api(ctx: Context, model: str | None = None) -> None:
    """Build, push, and deploy the API Docker container to Cloud Run."""
    command = "./scripts/deploy_api_cloudrun.sh"
    if model is not None:
        command = f"MODEL_NAME={quote(model)} {command}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task(name="deploy-bento")
def deploy_bento(ctx: Context) -> None:
    """Build, push, and deploy the BentoML API Docker container to Cloud Run."""
    ctx.run("./scripts/deploy_bento_cloudrun.sh", echo=True, pty=not WINDOWS)


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


@task
def dataset_statistics_check(ctx: Context) -> None:
    """running the dataset statistics (for automation via GitHub with cml_data.yaml)"""
    ctx.run(f"uv run src/{PROJECT_NAME}/dataset.py dataset-statistics", echo=True, pty=not WINDOWS)


@task(name="create-datadrift-reference")
def create_datadrift_reference(
    ctx: Context,
    bucket: str = "mlops-street-signs-prod-data",
) -> None:
    """Generate reference image features for data drift monitoring."""
    ctx.run(
        f"MONITORING_BUCKET={quote(bucket)} uv run src/{PROJECT_NAME}/monitoring/reference_features.py",
        echo=True,
        pty=not WINDOWS,
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


@task(name="start-local-api", aliases=("start-API",), auto_shortflags=False)
def start_local_API(ctx: Context, model: str | None = None, reload: bool = False, port: int = 8000) -> None:
    """Start API, can be viewed under http://localhost:8000/docs/"""
    command = f"uv run uvicorn --port {port} street_sign_project.fast_api:app"
    if reload:
        command = f"{command} --reload --reload-dir src/street_sign_project"
    if model is not None:
        command = f"MODEL_NAME={quote(model)} {command}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task(name="start-local-frontend", aliases=("start-local-streamlit",), auto_shortflags=False)
def start_local_frontend(
    ctx: Context,
    api_url: str = "http://localhost:8000",
    port: int = 8501,
    start_api: bool = True,
    api_reload: bool = False,
    model: str | None = None,
) -> None:
    """Start the Streamlit frontend for image upload and prediction display.
    has to be started via uvx because otherwise there is a dependency conflict with FastAPI"""
    if start_api:
        parsed = urlparse(api_url)
        api_host = parsed.hostname or "localhost"
        api_port = parsed.port or 8000
        if api_host in {"localhost", "127.0.0.1"}:
            if _is_port_open("127.0.0.1", api_port):
                logger.info(f"Local API already running on {api_host}:{api_port}.")
            else:
                _start_local_api_detached(ctx, port=api_port, model=model, reload=api_reload)
        else:
            logger.warning(f"start_api=True ignored because api_url points to non-local host '{api_host}'.")

    command = (
        f"API_URL={quote(api_url)} uvx --with requests --from streamlit streamlit run src/{PROJECT_NAME}/streamlit_app.py "
        f"--server.port {port}"
    )
    ctx.run(command, echo=True, pty=not WINDOWS)


@task(name="deploy-frontend", auto_shortflags=False)
def deploy_frontend(
    ctx: Context,
    api_service_name: str | None = None,
    api_url: str | None = None,
) -> None:
    """Build, push, and deploy the Streamlit frontend to Cloud Run."""
    command = "./scripts/deploy_frontend_cloudrun.sh"
    if api_service_name is not None:
        command = f"API_SERVICE_NAME={quote(api_service_name)} {command}"
    if api_url is not None:
        command = f"API_URL={quote(api_url)} {command}"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task(name="deploy-api-finn")
def deploy_api_finn(ctx: Context) -> None:
    """Build, push, and deploy the API Docker container to Cloud Run."""
    command = "PROJECT_ID=streetsignproject IMAGE_NAME=street-sign-api-finn SERVICE_NAME=street-sign-api-finn MODEL_NAME=YOLO_eps200_bs16_lr0.005_fr5_x.pt MONITORING_BUCKET=monitoring_bucket_finn ./scripts/deploy_api_cloudrun.sh"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task(name="stop-api-finn")
def stop_api_finn(ctx: Context) -> None:
    """Stop the API in Cloud Run."""
    command = "gcloud run services delete street-sign-api-finn --region=europe-west3 --project=streetsignproject"
    ctx.run(command, echo=True, pty=not WINDOWS)


@task(name="stop-frontend-cloud")
def stop_frontend_cloud(ctx: Context) -> None:
    """Stop the Streamlit frontend in Cloud Run."""
    command = "gcloud run services delete street-sign-frontend --region=europe-west3 --project=streetsignproject"
    ctx.run(command, echo=True, pty=not WINDOWS)
