# Author: Finn Schmidt, Editor: Kenny Kubsch
# Defining wrapper classes for the model for easier application later on
# Documentation on YOLO Website: https://docs.ultralytics.com/reference/engine/model#ultralytics.engine.model.Model.train
# Detailed Documentation on GitHub: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py    or .../predictior.py

from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
import typer
import yaml
from loguru import logger
from ultralytics import YOLO, settings
from ultralytics.engine.results import Results  # for Typing
from ultralytics.utils.metrics import DetMetrics  # for Typing

import wandb
from street_sign_project.evaluate import evaluate_tp_fp_fn
from street_sign_project.utils import project_root

MODELS_QUALITY_YAML = project_root() / "models" / "models_quality.yaml"
TEST_IMAGES = project_root() / "data" / "preprocessed" / "test" / "images"
TEST_LABELS = project_root() / "data" / "preprocessed" / "test" / "labels"


settings.update({"wandb": False})  # schalte das wandb logging von ultralytics ab, da wir das ja selber machen wollen

app = typer.Typer(no_args_is_help=True)

model_sizes = {"n", "s", "m", "l", "x"}
# For Typing, Define optimizer options
optimizer_options = Literal["auto", "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"]


class YOLOv26:
    """
    Wrapper für YOLO Version 2026 zur Segmentierung und Klassifizierung
    Model size n is default for cpu usage. For better performance run s or higher
    """

    def __init__(self, local_model_name: str | None = None, model_size: str = "n") -> None:
        """
        function for initializing a model object based on either a pretrained model (e.g. yolo26n.pt) or a locally saved model

        Params:
        local_model_name: if not none, a local model from the models folder will be loaded. if none, a pretrained model will be loaded
        model_size: size of pretrained model
        """
        if local_model_name is None or local_model_name == "None":
            self.model_name = None
            self.model_size = model_size
            if self.model_size not in model_sizes:
                logger.error("Not a valid model_size. Valid are: n, s, m, l, x. Reverting to default n.")
                self.model_size = "n"
            self.model = YOLO(f"yolo26{self.model_size}.pt")
            logger.info(f"Model yolo26{self.model_size}.pt loading completed")
        else:
            if not local_model_name.endswith(".pt"):
                raise ValueError('model name needs to be a ".pt" file')
            self.model_name = local_model_name
            model_path = project_root() / "models" / self.model_name
            if not model_path.exists():
                raise ValueError("model does not exist in the models folder")
            self.model = YOLO(str(model_path))
            logger.info(f"Model {self.model_name} loading completed")

    def train(
        self,
        data: str | Path | None,
        model_path: str | Path | None,
        epochs: int,
        batch_size: int,
        seed: int,
        optimizer: optimizer_options,
        lr0: float,
        freeze: int,
        device: Literal["auto", "cuda", "cpu"],
        workers: int,
        wb_entity: str,
        wb_project: str,
        wb_mode: Literal["online", "offline", "disabled", "shared"] | None,
        wb_dir: str | Path,
        patience: int,
    ) -> DetMetrics | None:
        """Fine-tune the YOLO model.

        Args:
            data: Path to the YOLO dataset YAML file.
            model_path: Path where the trained model should be saved.
            epochs: Number of training epochs.
            batch_size: Batch size used during training.
            seed: Random seed used by Ultralytics training.
            optimizer: Optimizer name passed to Ultralytics.
            lr0: Initial learning rate.
            freeze: Number of layers to freeze during training.
            device: Training device. Use "auto" to select CUDA when available and CPU otherwise.
            workers: Maximum number of dataloader workers.
            wb_entity: W&B entity used for experiment tracking.
            wb_project: W&B project used for experiment tracking.
            wb_mode: W&B mode, such as "online", "offline", "disabled", or "shared".
            wb_dir: Local directory used by W&B for run files.
        """
        if (
            not MODELS_QUALITY_YAML.exists()
        ):  # checken, dass die Models_quality_yaml zur Überwchung der Modellgüte existiert
            logger.critical(
                "models_quality.yaml doesn't exist. Create by running >>uv run invoke create-models-quality-yaml<<"
            )
            raise FileNotFoundError(
                "models_quality.yaml doesn't exist. Create by running >>uv run invoke create-models-quality-yaml<<"
            )
        if self.model_name is None:
            name_string = f"YOLO_eps{epochs}_bs{batch_size}_lr{lr0}_fr{freeze}_{self.model_size}"
            logger.info(f"Initialized training procedure based on yolo26{self.model_size}.pt")
            logger.info(f"Model is called {name_string}.pt")
        else:
            model_name_parts = self.model_name.split("_")
            base_model_eps = str([p for p in model_name_parts if p.startswith("eps")][0]).replace("eps", "")
            base_model_bs = str([p for p in model_name_parts if p.startswith("bs")][0]).replace("bs", "")
            base_model_lr = str([p for p in model_name_parts if p.startswith("lr")][0]).replace("lr", "")
            base_model_fr = str([p for p in model_name_parts if p.startswith("fr")][0]).replace("fr", "")
            base_model_size = model_name_parts[-1].split(".")[0]
            name_string = f"YOLO_eps{epochs}_bs{batch_size}_lr{lr0}_fr{freeze}_based_on_{base_model_eps}_{base_model_bs}_{base_model_lr}_{base_model_fr}_{base_model_size}"
            logger.info(f"Initialized training procedure based on {self.model_name}")
            logger.info(f"Model is called {name_string}.pt")

        self.name_string = name_string

        log_dir = project_root() / "reports" / "logs" / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"training_{name_string}.log"

        log_file = None

        def logger_callback(
            trainer,
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
            data = project_root() / "data" / "dataset.yaml"

        logger.info(
            f"Start Training {name_string} at {datetime.now()}"
        )  # outputs only to console because the file is resetted with start of training
        start = datetime.now()

        # Determine Device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device {device}")

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
            workers=workers,
            save_period=-1,
            exist_ok=True,
            patience=patience,
        )

        if train_results is None:
            raise ValueError("Training didn't work. Train results are None")

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
                "train_mAP50_detect": train_results.results_dict[
                    "metrics/mAP50(B)"
                ],  # metrics within yolo are for detection; there are no separate metrics for segmentation or classification
                "miss_rate_detect": 1 - train_results.results_dict["metrics/recall(B)"],
                "train_precision_detect": train_results.results_dict["metrics/precision(B)"],
                "box loss per epoch": box_epoch_loss,
                "cls loss per epoch": cls_epoch_loss,
                "box loss per epoch plot": wandb.Image(box_epoch_loss_plot),
                "cls loss per epoch plot": wandb.Image(cls_epoch_loss_plot),
            }
        )

        ##### Saving
        if model_path is None:
            save_dir = project_root() / "models"
            model_path = save_dir / f"{name_string}.pt"
        else:
            model_path = Path(model_path)

        logger.info(f"Saving model locally to {model_path}")
        self.save_model(file_path=model_path)

        artifact = wandb.Artifact(
            name=name_string, type="model", description=f"YOLO Model trained for {epochs} epochs."
        )
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)

        #####
        # hier wird jetzt versucht, dass immer die besten 3 Modelle gespeichert werden
        # dafür das neue Modell mit der korrekten Models quality.yaml abgleichen
        with open(str(MODELS_QUALITY_YAML), "r") as f:
            AP_dict = yaml.safe_load(f)
        tp_global, fp_global, fn_global = 0, 0, 0
        for image in TEST_IMAGES.glob("*.jpg"):
            label_path = TEST_LABELS / f"{image.stem}.txt"
            labels = []
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        labels.append([float(x) for x in line.strip().split()])
            preds = self.predict(image)
            tp, fp, fn = evaluate_tp_fp_fn(preds, labels, iou_threshold=0.5)
            tp_global += tp
            fp_global += fp
            fn_global += fn
        if (tp_global + fp_global) == 0:
            precision_50: float = 0
        else:
            precision_50: float = tp_global / (
                tp_global + fp_global
            )  # vereinfachte, da normale AP50 ein Mittelwert über alle Konfidenzschwellen ist
        AP_dict[f"{name_string}.pt"] = precision_50
        with open(str(MODELS_QUALITY_YAML), "w") as f:
            yaml.safe_dump(AP_dict, f)

        ############ Automatisches Löschen schlechter Modelle auskommentiert, da unnötig

        #        current_models = list(AP_dict.keys())
        #        current_values = list(AP_dict.values())
        #        min_idx = int(torch.tensor(current_values + [precision_50]).argmin().item())
        #        if min_idx != 3:
        #            if len(current_models) == 3:
        #                worst_model = current_models[min_idx]
        #                worst_model_path = project_root() / "models" / worst_model
        #                logger.info(
        #                    f"Trained model has precision_50 of {precision_50}. Old Model {worst_model} is the worst model with precision_50 of {current_values[min_idx]} and is deleted"
        #                )
        #                if worst_model_path.exists():
        #                    worst_model_path.unlink()
        #                else:
        #                    raise ValueError("Model to be deleted is not existing in the models folder.")
        #                del AP_dict[worst_model]  # Altes Modell aus AP_dict löschen
        #            else:
        #                logger.info("Less than 3 models saved, so deleting none.")
        #
        #            AP_dict[f"{name_string}.pt"] = precision_50
        #            with open(str(MODELS_QUALITY_YAML), "w") as f:
        #                yaml.safe_dump(AP_dict, f)
        #
        #        else:
        #            logger.info(f"Trained model has precision_50 of {precision_50} and is therefore not saved")

        return train_results

    def predict(self, new_data: str | Path | None = None, conf: float = 0.25) -> Results:
        """
        method for predicting with the model

        params:
            new_data: file path
            conf: confidence score threshold for prediction
        """
        if new_data is None or not str(new_data).endswith((".yaml", ".jpg")):
            raise RuntimeError(
                "Keine Daten für prediction angegeben. Datenverweis muss als .yaml oder .jpg gegeben werden"
            )
        # Modell wird automatisch in Evaluationsmodus gesetzt
        # Data preprocessing für daten im Format (N, 3, H, W) nicht nötig
        return self.model.predict(
            source=new_data, conf=conf
        )[
            0
        ]  # Ergebnis nur für das erste Bild das reingesteckt wurde, deswegen [0], wir wenden es ja eh immer nur auf ein Bild an

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

    def evaluate(self) -> float:
        """vereinfachte, da normale AP50 ein Mittelwert über alle Konfidenzschwellen ist"""
        tp_global, fp_global, fn_global = 0, 0, 0
        for image in TEST_IMAGES.glob("*.jpg"):
            label_path = TEST_LABELS / f"{image.stem}.txt"
            labels = []
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        labels.append([float(x) for x in line.strip().split()])
            preds = self.predict(image)
            tp, fp, fn = evaluate_tp_fp_fn(preds, labels, iou_threshold=0.5)
            tp_global += tp
            fp_global += fp
            fn_global += fn
        if (tp_global + fp_global) == 0:
            precision_50: float = 0
        else:
            precision_50: float = tp_global / (tp_global + fp_global)
        return precision_50

    def cleanup_savings(self) -> bool:
        """
        Abgleich mit models_quality.yaml, um die 3 besten Modelle zu speichern
        returns a bool that indicates whether model should be saved or not
        """
        precision_50 = self.evaluate()

        with open(str(MODELS_QUALITY_YAML), "r") as f:
            AP_dict = yaml.safe_load(f)
        current_models = list(AP_dict.keys())
        current_values = list(AP_dict.values())
        current_n = len(current_models)
        min_idx = int(torch.tensor(current_values + [precision_50]).argmin().item())
        if min_idx != current_n:
            if current_n >= 3:
                worst_model = current_models[min_idx]
                worst_model_path = project_root() / "models" / worst_model
                logger.info(
                    f"Trained model has precision_50 of {precision_50}. Old Model {worst_model} is the worst model with precision_50 of {current_values[min_idx]} and is deleted"
                )
                if worst_model_path.exists():
                    worst_model_path.unlink()
                else:
                    raise ValueError("Model to be deleted is not existing in the models folder.")
                del AP_dict[worst_model]  # Altes Modell aus AP_dict löschen
            else:
                logger.info("Less than 3 models saved, so deleting none.")

            AP_dict[self.name_string] = precision_50
            with open(str(MODELS_QUALITY_YAML), "w") as f:
                yaml.safe_dump(AP_dict, f)
            saving_model = True
        else:
            logger.info(f"Trained model has precision_50 of {precision_50} and is therefore not saved")
            saving_model = False
        return saving_model
