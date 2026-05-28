# Author: Finn Schmidt
# Defining wrapper classes for the model for easier application later on
# Documentation on YOLO Website: https://docs.ultralytics.com/reference/engine/model#ultralytics.engine.model.Model.train
# Detailed Documentation on GitHub: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py    or .../predictior.py

import typer
from ultralytics import YOLO
from loguru import logger
from pathlib import Path
from datetime import datetime

app = typer.Typer(no_args_is_help=True)

model_sizes = {"n", "s", "m", "l", "x"}

def project_root() -> Path:
    """Finds parent folder where pyproject.toml lies"""
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found")


class YOLOv26():
    """
    Wrapper für YOLO Version 2026 zur Segmentierung und Klassifizierung
    Model size n is default for cpu usage. For better performance run s or higher
    """
    def __init__(self, model_size: str = "n"):
        self.model_size = model_size
        if self.model_size not in model_sizes:
            logger.error("Not a valid model_size. Valid are: n, s, m, l, x. Reverting to default n.")
            self.model_size = "n"
        self.model = YOLO(f"yolo26{self.model_size}.pt")
        logger.info(f"Model yolo26{self.model_size}.pt loading completed")

    def train(self, data: str | Path | None = None, model_path: str | None = None, epochs: int = 1, batch_size: int = 16, lr0:float = 0.005, freeze:int=10, device = "cpu"):
        """
        method for fine-tuning the Yolo model

        params:

        data: path to yaml file
        model_name: under which name it is saved
        epochs: number of epochs
        batch_size: -""-
        lr0: initial learning rate
        device: "cuda" or "cpu"
        """
        if data is None:
            root = project_root() 
            data = root / "data" / "dataset.yaml"

        logger.info(f"Start Training yolo26{self.model_size}_ep{epochs}_bs{batch_size}_fr{freeze} at {datetime.now()}")
        train_results=self.model.train(data=str(data), epochs=epochs, freeze=freeze, batch=batch_size, lr0 = lr0, device=device, save_period=-1, project = None, name = None)  # save_period to not save initial model and project & name to not save the runs/ folder
        logger.info(f"Training results: {train_results}")
        logger.info(f"Finished training yolo26{self.model_size}_ep{epochs}_bs{batch_size}_fr{freeze} at {datetime.now()}")

        # saving
        if model_path is None:
            root = project_root()
            model_path = root / "models" / f"YOLO_eps{epochs}_bs_{batch_size}_lr{lr0}_fr{freeze}.pt"
        self.save_model(model_path)
        logger.info(f"Model saved under {model_path}")

        return train_results

    def predict(self, new_data: str | Path | None = None, conf:float=0.25):
        """
        method for predicting with the model

        params:
            new_data: file path
        """
        if new_data is None or not str(new_data).endswith(".yaml"):
            raise RuntimeError("Keine Daten für prediction angegeben. Datenverweis muss als .yaml gegeben werden")
        # Modell wird automatisch in Evaluationsmodus gesetzt 
        # Data preprocessing für daten im Format (N, 3, H, W) nicht nötig
        return self.model.predict(source=new_data, conf=conf)

    def test(self, data:str | Path):
        """
        Tests Model on test data and returns metrics

        params:
            data: path to yaml file
        """
        if not str(data).endswith(".yaml"):
            raise RuntimeError("Keine Daten zum Testen angegeben. Datenverweis muss als .yaml gegeben werden")
        return self.model.val(data=str(data))

    def save_model(self, file_path: str | Path):
        """
        method for saving a model under a given path as a .pt file
        """
        if not str(file_path).endswith(".pt"):
            raise ValueError("file path needs to be a \".pt\" file")
        self.model.save(filename = file_path)


def train_model():
    model = YOLOv26()
    results = model.train()
    print(f"train results: \n {results}")
