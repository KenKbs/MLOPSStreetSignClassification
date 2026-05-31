# Author: Finn Schmidt
# Defining wrapper classes for the model for easier application later on
# Documentation on YOLO Website: https://docs.ultralytics.com/reference/engine/model#ultralytics.engine.model.Model.train
# Detailed Documentation on GitHub: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py    or .../predictior.py

import typer
from typing import Literal
from ultralytics import YOLO
from loguru import logger
from pathlib import Path
from datetime import datetime
import wandb
import matplotlib.pyplot as plt

from ultralytics import settings
from ultralytics.engine.results import Results  # for Typing
from ultralytics.utils.metrics import DetMetrics  # for Typing

settings.update({"wandb": False})  # schalte das wandb logging von ultralytics ab, da wir das ja selber machen wollen

app = typer.Typer(no_args_is_help=True)

model_sizes = {"n", "s", "m", "l", "x"}
# For Typing, Define optimizer options
optimizer_options = Literal["auto", "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"]


def project_root() -> Path:
    """Finds parent folder where pyproject.toml lies"""
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found")


class YOLOv26:
    """
    Wrapper für YOLO Version 2026 zur Segmentierung und Klassifizierung
    Model size n is default for cpu usage. For better performance run s or higher
    """

    def __init__(self, model_size: str = "n") -> None:
        self.model_size = model_size
        if self.model_size not in model_sizes:
            logger.error("Not a valid model_size. Valid are: n, s, m, l, x. Reverting to default n.")
            self.model_size = "n"
        self.model = YOLO(f"yolo26{self.model_size}.pt")
        logger.info(f"Model yolo26{self.model_size}.pt loading completed")

    def train(
        self,
        data: str | Path | None = None,
        model_path: str | Path | None = None,
        epochs: int = 10,
        batch_size: int = 16,
        seed: int = 420,
        optimizer: optimizer_options = "auto",
        lr0: float = 0.005,
        freeze: int = 10,
        device: str = "cpu",
        wb_entity: str = "k-kubsch-ludwig-maximilian-university-of-munich",
        wb_project: str = "StreetSignClassification",
        wb_mode: Literal["online", "offline", "disabled", "shared"] | None = None,
        wb_dir: str | Path = "/tmp",
    ) -> DetMetrics | None:
        """
        method for fine-tuning the Yolo model

        params:

        data: path to yaml file
        model_path: under which name it is saved
        epochs: number of epochs
        batch_size: batch size
        lr0: initial learning rate
        freeze: number of layers in which the parameters are frozen and not trained
        device: "cuda" or "cpu"
        """
        name_string = f"YOLO_eps{epochs}_bs_{batch_size}_lr{lr0}_fr{freeze}"

        log_dir = project_root() / "reports" / "logs" / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"training_{name_string}.log"

        log_file = None

        def logger_callback(
            trainer
        ):  # this function is needed so that with the start of the training, YOLO doesn't block the saving of our log file
            nonlocal log_file
            log_file = logger.add(str(log_path), level="DEBUG", rotation="10 MB")

        self.model.add_callback("on_train_start", logger_callback)

        box_epoch_loss = []
        cls_epoch_loss = []

        def epoch_metrics_callback(trainer):  # function to extract the loss after every epoch
            box_epoch_loss.append(float(trainer.tloss[0]))
            cls_epoch_loss.append(float(trainer.tloss[1]))

        self.model.add_callback("on_train_epoch_end", epoch_metrics_callback)

        wandb.init(
            entity=wb_entity,
            project=wb_project,
            dir=wb_dir,
            mode=wb_mode,
            config={"lr": lr0, "batch_size": batch_size, "epochs": epochs, "freeze": freeze},
        )
        if data is None:
            root = project_root()
            data = root / "data" / "dataset.yaml"

        logger.info(
            f"Start Training {name_string} at {datetime.now()}"
        )  # outputs only to console because the file is resetted with start of training
        start = datetime.now()

        # training
        train_results = self.model.train(
            data=str(data),
            epochs=epochs,
            seed=seed,
            freeze=freeze,
            batch=batch_size,
            optimizer=optimizer,
            lr0=lr0,
            device=device,
            save_period=-1,
            exist_ok=True,
        )

        # logging
        logger.info(f"Finished training {name_string} at {datetime.now()}; time difference of {datetime.now() - start}")
        logger.info(f"Training results: {train_results.results_dict}")
        logger.info(f"box loss per epoch: {box_epoch_loss}")
        logger.info(f"cls loss per epoch: {cls_epoch_loss}")

        # plots
        x_box, y_box = list(range(1, len(box_epoch_loss) + 1)), box_epoch_loss
        x_cls, y_cls = list(range(1, len(cls_epoch_loss) + 1)), cls_epoch_loss

        box_epoch_loss_plot, object = plt.subplots()
        object.plot(x_box, y_box, marker="o")
        object.set_xlabel("epoch")
        object.set_ylabel("loss")
        object.set_title("Segmentation Loss")
        object.grid(True)

        cls_epoch_loss_plot, object2 = plt.subplots()
        object2.plot(x_cls, y_cls, marker="o")
        object2.set_xlabel("epoch")
        object2.set_ylabel("loss")
        object2.set_title("Segmentation Loss")
        object2.grid(True)

        # wandb
        wandb.log(
            {
                "train_mAP50": train_results.results_dict["metrics/mAP50(B)"],
                "train_precision": train_results.results_dict["metrics/precision(B)"],
                "box loss per epoch": box_epoch_loss,
                "cls loss per epoch": cls_epoch_loss,
                "box loss per epoch plot": wandb.Image(box_epoch_loss_plot),
                "cls loss per epoch plot": wandb.Image(cls_epoch_loss_plot),
            }
        )

        # saving
        if model_path is None:
            root = project_root()
            model_path = root / "models" / f"{name_string}.pt"
        self.save_model(model_path)
        logger.info(f"Model saved under {model_path}")
        if log_file is not None:
            logger.remove(log_file)

        return train_results

    def predict(self, new_data: str | Path | None = None, conf: float = 0.25) -> list[Results]:
        """
        method for predicting with the model

        params:
            new_data: file path
            conf: confidence score threshold for prediction
        """
        if new_data is None or not str(new_data).endswith(".yaml"):
            raise RuntimeError("Keine Daten für prediction angegeben. Datenverweis muss als .yaml gegeben werden")
        # Modell wird automatisch in Evaluationsmodus gesetzt
        # Data preprocessing für daten im Format (N, 3, H, W) nicht nötig
        return self.model.predict(source=new_data, conf=conf)

    def test(self, data: str | Path) -> DetMetrics:
        """
        Tests Model on test data and returns metrics

        params:
            data: path to yaml file
        """
        if not str(data).endswith(".yaml"):
            raise RuntimeError("Keine Daten zum Testen angegeben. Datenverweis muss als .yaml gegeben werden")
        return self.model.val(data=str(data))

    def save_model(self, file_path: str | Path) -> None:
        """
        method for saving a model under a given path as a .pt file
        """
        if not str(file_path).endswith(".pt"):
            raise ValueError('file path needs to be a ".pt" file')
        self.model.save(filename=file_path)


def train_model() -> None:
    model = YOLOv26()
    results = model.train()
    print(f"train results: \n {results}")
