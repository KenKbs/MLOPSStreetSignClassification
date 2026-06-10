from pathlib import Path
from typing import Optional

import cv2
import typer
from loguru import logger

from street_sign_project.model import YOLOv26
from street_sign_project.utils import project_root

app = typer.Typer()

TEST_IMAGES = project_root() / "data" / "preprocessed" / "test" / "images"


@app.command()
def plot_image_pred(model_name: Optional[str] = None, image_folder_path: Optional[Path] = None) -> None:
    if image_folder_path is None or not Path(image_folder_path).exists():
        logger.info("Reverting to test images")
        image_folder_path = TEST_IMAGES
    if model_name is None:
        logger.info("reverting back to YOLO_eps100_bs8_lr0.005_fr10_x.pt")
        model_name = "YOLO_eps100_bs8_lr0.005_fr10_x.pt"
    print_counter = 0
    model = YOLOv26(model_name)
    for image_path in Path(image_folder_path).glob("*.jpg"):
        if print_counter >= 5:
            break
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Bild {image_path} konnte nicht geladen werden")
            continue
        pred = model.predict(image_path)
        if pred.boxes is None:
            raise ValueError("pred.boxes is None, but should be a list of prediction objects")
        for box in pred.boxes:
            xyxy = box.xyxy[0]
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            label = f"{int(box.cls[0].item())}: {round(box.conf[0].item(), 2)}"
            cv2.rectangle(
                image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[0] + 80), int(xyxy[1] + 12)), (0, 0, 0), -1
            )  # schwarzer hintergrund
            cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1] + 12)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        hinweis = "press any key for next image or ctrl+z to exit"
        cv2.rectangle(image, (0, 0), (500, 15), (255, 255, 255), -1)
        cv2.putText(image, hinweis, (0, 12), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
        cv2.imshow("Yolo Detection", image)
        cv2.waitKey(0)
        print_counter += 1
    cv2.destroyAllWindows()


@app.command()
def Lueckenfüller():
    print("Lückenfüller, da typer mehrere app commands pro Datei braucht")


if __name__ == "__main__":
    app()
