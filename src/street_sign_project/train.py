import hydra
from omegaconf import DictConfig

from street_sign_project.model import YOLOv26


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train_model(config: DictConfig):
    """Main train model Orchestrator, CLI wrapped by hydra"""
    # Config path is realative to this function where we inject hydra
    # Get config branches
    path_cfg = config.paths
    model_cfg = config.model
    train_cfg = config.training
    wandb_cfg = config.wandb
    model = YOLOv26(model_size=model_cfg.yolo_model_size)
    results = model.train(
        data=path_cfg.data_yaml,
        model_path=path_cfg.model_path,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
        optimizer=train_cfg.optimizer,
        lr0=train_cfg.lr0,
        freeze=train_cfg.freeze,
        device=train_cfg.device,
        workers = train_cfg.workers,
        wb_entity=wandb_cfg.entity,
        wb_project=wandb_cfg.project,
        wb_mode=wandb_cfg.mode,
        wb_dir=wandb_cfg.dir,
    )

    print(f"train results: \n {results}")


if __name__ == "__main__":
    train_model()  # Stopped using typer, because Hydra becomes CLI manager here!
