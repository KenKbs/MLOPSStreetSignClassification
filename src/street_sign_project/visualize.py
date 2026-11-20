"""Draw dataset annotations for inspection and dataset-statistics reports."""

from pathlib import Path
from typing import Optional

import cv2
from loguru import logger

from street_sign_project.data import DatasetItem


def plot_image_label(
    dataset_items: list[DatasetItem], true_or_pred: bool = True, save_dir: Optional[str] = None
) -> None:
    """Draw dataset bounding boxes and save or display the annotated images.

    Args:
        dataset_items: Images and their corresponding YOLO label files.
        true_or_pred: Whether to caption annotations as ground truth or predictions.
        save_dir: Output directory. When omitted, display images in desktop windows.
    """
    for i, dataset_item in enumerate(dataset_items):
        label_path = dataset_item.label_path
        image_path = dataset_item.image_path
        if label_path is not None:
            if not Path(image_path).stem == label_path.stem:
                logger.warning(f"Bild und Label an index {i} stimmen nicht überein. Suche nach korrektem Label...")
                for label_p in label_path.parent.glob("*.txt"):
                    if Path(image_path).stem == label_p.stem:
                        label_path = label_p
                        logger.info("Correct label path found.")
                        break
        if not Path(image_path).exists():
            logger.error(f"Image path {image_path} does not exist")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Bild {image_path} konnte nicht geladen werden")
            continue
        img_h, img_w, _ = image.shape
        boxes: list[list[float]] = []
        classes = []
        if label_path is not None:
            with open(label_path, "r") as label_file:
                for label_line in label_file.readlines():
                    label_split = label_line.strip().split()
                    if not label_split:
                        continue
                    boxes.append([float(x) for x in label_split[1:5]])
                    classes.append(int(label_split[0]))
        if boxes == []:
            logger.info(f"no label box found for image {image_path}")
        else:
            for j, box in enumerate(boxes):
                box[0] = box[0] * img_w
                box[1] = box[1] * img_h
                box[2] = box[2] * img_w
                box[3] = box[3] * img_h
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
                    (int(box[0] - 0.5 * box[2] + 20), int(box[1] - 0.5 * box[3] + 12)),
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
        if save_dir is not None:
            Path(save_dir).mkdir(exist_ok=True)
            output_path = Path(save_dir) / f"labels_{Path(image_path).name}"
            cv2.imwrite(str(output_path), image)
        else:
            cv2.imshow("Yolo Detection", image)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
