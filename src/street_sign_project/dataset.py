from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import typer
from cv2.typing import MatLike
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset

from street_sign_project.data import DatasetItem, _collect_dataset_items
from street_sign_project.utils import project_root
from street_sign_project.visualize import plot_image_label

app = typer.Typer()


def _get_label_classes(file_label_paths: list[Path]) -> Tensor:
    """
    extract class labels from label files to plot distributions
    Input: Paths to label files
    Output Tensor with all class IDs
    """
    classes = []
    for path in file_label_paths:
        with path.open("r", encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                values = line.strip().split()
                if not values:
                    continue
                if len(values) != 5:
                    raise ValueError(f"Expected 5 YOLO values in {path}:{line_number}, got {len(values)} entries.")
                classes.append(int(values[0]))
    classes_tensor = torch.tensor(classes, dtype=torch.long)
    if classes_tensor is None:
        raise ValueError("Return argument of _get_label_classes was not calculated correctly")
    return classes_tensor


class YoloDataSet(Dataset):
    """
    Dataset class is needed for module M19: Continuous Machine Learning

    Params:

    """

    name: str = "YoloDataSet"

    def __init__(self, split: str = "train") -> None:
        if split not in ("train", "test", "valid"):
            raise ValueError("Split does not exist")
        self.data_dir = Path(f"{project_root()}/data/preprocessed/{split}")
        self.image_dir = self.data_dir / "images"
        self.image_paths = list(self.image_dir.glob("*.jpg"))
        self.label_dir = self.data_dir / "labels"
        self.label_paths = list(self.label_dir.glob("*.txt"))
        self.items = _collect_dataset_items(self.image_dir, self.label_dir)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> tuple[MatLike, dict[str, Tensor]]:
        item = self.items[index]
        image = cv2.imread(str(item.image_path))
        if image is None:
            raise ValueError(f"Image {str(item.image_path)} could not be loaded")

        if item.label_path is None:
            return image, {"Boxes": torch.tensor([]), "classes": torch.tensor([])}

        boxes = []
        classes = []
        label_path = item.label_path
        if not Path(item.image_path).stem == Path(label_path).stem:
            logger.warning(f"Bild und Label an index {index} stimmen nicht überein. Suche nach korrektem Label...")
            for label_p in Path(item.label_path).parent.glob("*.txt"):
                if Path(item.image_path).stem == label_p.stem:
                    label_path = label_p
                    logger.info("Correct label path found.")
                    break
                raise ValueError(
                    f"Correct label for {Path(item.image_path).name} not found in {Path(item.label_path).parent}"
                )

        if label_path is not None and label_path.exists():
            with open(label_path, "r") as labels:
                for label in labels.readlines():
                    label_split = label.strip().split()
                    if label_split[4] is None or not label_split:
                        continue
                    boxes.append([float(x) for x in label_split[1:5]])
                    classes.append(int(label_split[0]))

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            classes_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            classes_tensor = torch.tensor(classes, dtype=torch.long)

        target = {"boxes": boxes_tensor, "classes": classes_tensor}

        return image, target


@app.command()
def dataset_statistics(datadir: str = "data") -> None:
    """Compute dataset statistics."""
    train_dataset = YoloDataSet(split="train")
    test_dataset = YoloDataSet(split="test")
    valid_dataset = YoloDataSet(split="valid")
    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"Test dataset: {test_dataset.name}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")
    print("\n")
    print(f"Validation dataset: {valid_dataset.name}")
    print(f"Number of images: {len(valid_dataset)}")
    print(f"Image shape: {valid_dataset[0][0].shape}")
    print(train_dataset.items[0].image_path)
    print(train_dataset.items[0].label_path)

    plot_image_label(train_dataset.items[:5], save_dir="plots")

    train_label_distribution = torch.bincount(_get_label_classes(train_dataset.label_paths), minlength=73)
    test_label_distribution = torch.bincount(_get_label_classes(test_dataset.label_paths), minlength=73)

    Path("plots/").mkdir(exist_ok=True)

    plt.bar(torch.arange(len(train_label_distribution)), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("plots/train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(len(test_label_distribution)), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("plots/test_label_distribution.png")
    plt.close()


@app.command()
def dummy_function() -> None:
    print("Dummy function because typer needs >1 commands per .py file")


if __name__ == "__main__":
    app()
