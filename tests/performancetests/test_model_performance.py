# test for staged wnadb models in github to see that the models can do inference fast enough for deployment

import os
import time

import wandb
from street_sign_project.model import YOLOv26
from torch import rand


def load_model(model_checkpoint):
    # greift über github umgebungsvariablen auf wandb zu. der wandb_api_key ist ein persönlicher access key und nicht vom team
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(
        root="./reports/logs/models"
    )  # lädt das Modell Artefakt aus wandb in den github ordner root zum Testen. Sobald der test fertig ist, wird das Artefakt wieder gelöscht
    file_name = artifact.files()[0].name
    return YOLOv26.load_from_checkpoint(f"./reports/logs/models/{file_name}")


def test_model_speed():
    model = load_model(
        os.getenv("MODEL_NAME")
    )  # der MODEL_NAME wird von der yaml Datei, die den Test einleitet automatisch abgefangen und in github als Umgebungsvariable angelegt
    start = time.time()
    i = 0
    while i < 10:
        model.model(
            rand(1, 3, 640, 640)
        )  # model.model ruft dann einfach das YOLO model Objekt in der YOLOv26 Klasse auf
        i += 1
    end = time.time()
    assert end - start < 10
