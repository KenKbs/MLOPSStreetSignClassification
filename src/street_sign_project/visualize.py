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
def plot_image_label(
    image_path_list: list[str] | list[Path], label_path_list: list[str] | list[Path], true_or_pred: bool = True
) -> None:
    for i, image_path in enumerate(image_path_list):
        if not Path(image_path).stem == Path(label_path_list[i]).stem:
            raise ValueError(f"Image and label at position {i} in the input don't have the same name")
        if not Path(image_path).exists():
            logger.error(f"Image path {image_path} does not exist")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Bild {image_path} konnte nicht geladen werden")
            continue
        boxes: list[list[float]] = []
        classes = []
        with open(label_path_list[i], "r") as label_file:
            for label_line in label_file.readlines():
                label_split = label_line.strip().split()
                if label_split[4] is None or not label_split:
                    continue
                boxes.append([float(x) for x in label_split[1:5]])
                classes.append(int(label_split[0]))
        if boxes == []:
            logger.info(f"no label box found for image {image_path}")
            continue
        for j, box in enumerate(boxes):
            cv2.rectangle(
                image,
                (int(box[0] - 0.5 * box[2]), int(box[1] - 0.5 * box[3])),
                (int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3])),
                (0, 255, 0),
                2,
            )
            label = f"{classes[j]}"
            cv2.rectangle(
                image,
                (int(box[0] - 0.5 * box[2]), int(box[1] - 0.5 * box[3])),
                (int(box[0] - 0.5 * box[2] + 80), int(box[1] - 0.5 * box[3] + 12)),
                (0, 0, 0),
                -1,
            )  # schwarzer hintergrund
            cv2.putText(
                image,
                label,
                (int(box[0] - 0.5 * box[2]), int(box[1] - 0.5 * box[3] + 12)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )
        hinweis = "press any key for next image or ctrl+z to exit"
        cv2.rectangle(image, (0, 0), (500, 15), (255, 255, 255), -1)
        cv2.putText(image, hinweis, (0, 12), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
        # noch sagen ob es prediction oder label ist
        if true_or_pred:
            hinweis_2 = "True Label"
        else:
            hinweis_2 = "Prediction"
        cv2.rectangle(image, (0, 15), (115, 30), (255, 255, 255), -1)
        cv2.putText(image, hinweis_2, (0, 27), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
        cv2.imshow("Yolo Detection", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
