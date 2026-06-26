# file for setting up the API access to our model's predictions

import re
from contextlib import asynccontextmanager
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import cv2
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from street_sign_project.model import YOLOv26
from street_sign_project.utils import project_root


@asynccontextmanager  # for doing something at start and end of app
async def lifespan(app: FastAPI):
    print("Starting Application")
    global model
    model = YOLOv26(local_model_name="YOLO_eps420_bs8_lr0.005_fr10_x.pt")
    yield
    print("Closing Application")
    del model


app = FastAPI(lifespan=lifespan)


@app.post(
    "/image_input/"
)  # uploading, changing an giving back an image, the "noqa: B008" Comment for ruff to not give an error
async def cv_model(data: UploadFile = File(...)):  # noqa: B008
    filecount = sum(1 for x in Path(f"{project_root()}/API_uploads/input").iterdir() if x.is_file())
    input_image_path = f"{project_root()}/API_uploads/input/image_{filecount}.jpg"
    output_image_path = f"{project_root()}/API_uploads/output/image_{filecount}.jpg"

    with open(input_image_path, "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError("Bild konnte nicht geladen werden")
    pred = model.predict(input_image_path)
    if pred.boxes is None:
        raise ValueError("pred.boxes is None, but should be a list of prediction objects")
    for box in pred.boxes:
        xyxy = box.xyxy[0]
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        lbl = f"{int(box.cls[0].item())}: {round(box.conf[0].item(), 2)}"
        cv2.rectangle(
            img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[0] + 80), int(xyxy[1] + 12)), (0, 0, 0), -1
        )  # schwarzer hintergrund
        cv2.putText(img, lbl, (int(xyxy[0]), int(xyxy[1] + 12)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imwrite(output_image_path, img)

    return FileResponse(output_image_path, media_type="image/jpeg")
