# Author: Finn Schmidt
# Defining wrapper classes for the model for easier application later on
# Documentation for YOLO under: https://docs.ultralytics.com/reference/engine/model#ultralytics.engine.model.Model.train


from ultralytics import YOLO
from loguru import logger
from pathlib import Path
from datetime import datetime

model_sizes = {"n", "s", "m", "l", "x"}

def project_root() -> Path:
    """Finds parent folder where pyproject.toml lies"""
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found")


class YOLOv26seg():
    """Wrapper für YOLO Version 2026 zur Segmentierung"""
    def __init__(self, model_size: str = "s"):
        self.model_size = model_size
        if self.model_size not in model_sizes:
            logger.error("Not a valid model_size. Valid are: n, s, m, l, x. Reverting to default s.")
            self.model_size = "s"
        self.model = YOLO(f"yolo26{self.model_size}-seg.pt")
        logger.info(f"Model yolo26{self.model_size}-seg.pt loading completed")

    def train(self, data: str | Path | None = None, model_name: str | None = None, epochs: int = 50, batch_size: int = 16, lr0:float = 0.005, freeze:int=10, device = "cpu"):
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
            data = root / "data" / "raw"   #TODO Muss noch angepasst werden und auf die yaml zeigen, wenn sie dann da ist
        if model_name is None:
            model_name = f"yolo26{self.model_size}-seg_ep{epochs}_bs{batch_size}_fr{freeze}"
        logger.info(f"Start Training yolo26{self.model_size}-seg_ep{epochs}_bs{batch_size}_fr{freeze} at {datetime.now()}")
        self.model.train(data=str(data), epochs=epochs, freeze=freeze, batch=batch_size, lr0 = lr0, name=model_name, device=device)
        logger.info(f"Finished training yolo26{self.model_size}-seg_ep{epochs}_bs{batch_size}_fr{freeze} at {datetime.now()}")

