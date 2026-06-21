import os
from typing import List, Optional

import typer
import wandb

app = typer.Typer()


@app.command()
def link_model(
    artifact_path: str,
    # noqa so that ruff understands the typer.Option
    aliases: Optional[List[str]] = typer.Option(None, "--aliases", "-a"),  # noqa: B008
) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with.

    Example:
        model_management link-model entity/project/artifact_name:version -a staging -a best

    """
    if aliases is None:
        aliases = ["staging"]

    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact_name_version = artifact_path.split("/")[-1]
    artifact_name, _ = artifact_name_version.split(":")

    target_registry_path = f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}"
    artifact = api.artifact(artifact_path)
    # artifact.link(target_path=target_registry_path, aliases=aliases)
    artifact.aliases = aliases
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {aliases}")


if __name__ == "__main__":
    app()
