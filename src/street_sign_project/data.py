import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from loguru import logger
from openpyxl import load_workbook

from src.street_sign_project.utils import project_root

# Define Path constants
DEFAULT_RAW_DATA_DIR = project_root() / "data" / "raw"

DEFAULT_PREPROCESS_DATA_DIR = project_root() / "data" / "preprocessed"
DEFAULT_CONFIG_DIR = project_root() / "configs"
DEFAULT_MAPPING_PATH_XLSX = DEFAULT_CONFIG_DIR / "street_sign_class_mapping.xlsx"
DEFAULT_MAPPING_PATH_CSV = DEFAULT_CONFIG_DIR / "street_sign_class_mapping.csv"
MAPPING_SHEET_NAME = "canonical_mapping"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Define Typing constants:
DatasetName = Literal["germany", "italy"]

app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True)
class ClassMapping:
    """Class names and dataset-specific class ID mappings."""

    class_names: tuple[str, ...]
    germany_to_canonical: dict[int, int]
    italy_to_canonical: dict[int, int]

    @classmethod
    def from_csv(cls, mapping_path: Path = DEFAULT_MAPPING_PATH_CSV) -> "ClassMapping":
        """Class constructor. Load class names and dataset-specific mappings from a CSV file."""
        class_names: list[str] = []
        germany_to_canonical: dict[int, int] = {}
        italy_to_canonical: dict[int, int] = {}

        logger.info(f"Loading class mapping from {mapping_path}")
        with mapping_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                canonical_id = int(row["canonical_id"].strip())
                canonical_name = row["canonical_name"].strip()

                while len(class_names) <= canonical_id:
                    class_names.append("")
                class_names[canonical_id] = canonical_name

                italy_id = row["italy_id"].strip()
                if italy_id:
                    italy_to_canonical[int(italy_id)] = canonical_id

                germany_id = row["germany_id"].strip()
                if germany_id and germany_id.isdecimal():
                    germany_to_canonical[int(germany_id)] = canonical_id

        logger.info(f"Loaded {len(class_names)} canonical classes")
        return cls(
            class_names=tuple(class_names),
            germany_to_canonical=germany_to_canonical,
            italy_to_canonical=italy_to_canonical,
        )

    def to_canonical(self, dataset_name: DatasetName, class_id: int) -> int:
        """Return the canonical ID for a dataset specific class-id"""
        if dataset_name == "germany":
            return self.germany_to_canonical[class_id]
        elif dataset_name == "italy":
            return self.italy_to_canonical[class_id]

        raise ValueError(f"Unknown dataset name: {dataset_name}")


def _remap_label_file(
    file_path: Path,
    output_path: Path,
    class_mapping: ClassMapping,
    dataset_name: DatasetName,
) -> None:
    """Remap one YOLO label file to canonical class IDs.

    Args:
        file_path: Path to the source label file.
        output_path: Path where the remapped label file should be written.
        class_mapping: Mapping used to convert dataset-specific class IDs.
        dataset_name: Source dataset name.

    Raises:
        ValueError: If a label line is malformed.
        KeyError: If a class ID has no canonical mapping.
    """
    remapped_lines: list[str] = []

    with file_path.open("r", encoding="utf-8") as label_file:
        # One file can have multiple lines
        for line_number, line in enumerate(label_file, start=1):
            values = line.strip().split()
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(f"Expected 5 YOLO values in {file_path}:{line_number}, got {len(values)}.")

            # Get class id and remap to canonical
            class_id = int(values[0])
            try:
                canonical_id = class_mapping.to_canonical(dataset_name, class_id)
            except KeyError as error:
                raise KeyError(f"No canonical mapping for {dataset_name} class ID {class_id}.") from error

            # Add remapped lines to a list of strings
            remapped_lines.append(" ".join([str(canonical_id), *values[1:]]))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remap a list of lines to one string
    output_text = "\n".join(remapped_lines)
    output_text += "\n"  # Add empty line to the end (convention)

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(output_text)


def _remap_label_folder(
    input_labels_dir: Path,
    output_labels_dir: Path,
    class_mapping: ClassMapping,
    dataset_name: DatasetName,
) -> None:
    """Remap all YOLO label files in one folder to canonical values"""
    logger.info(f"Remapping {dataset_name} labels from {input_labels_dir} to {output_labels_dir}")

    remapped_count = 0
    for label_path in input_labels_dir.glob("*.txt"):
        output_path = output_labels_dir / label_path.name
        _remap_label_file(
            file_path=label_path,
            output_path=output_path,
            class_mapping=class_mapping,
            dataset_name=dataset_name,
        )
        remapped_count += 1

    logger.info(f"Remapped {remapped_count} label files for {dataset_name}")


def _copy_image_folder(input_images_dir: Path, output_images_dir: Path) -> None:
    """Copy all image files in one folder to another folder."""
    logger.info(f"Copying images from {input_images_dir} to {output_images_dir}")
    output_images_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for image_path in input_images_dir.iterdir():
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        shutil.copy2(image_path, output_images_dir / image_path.name)
        copied_count += 1

    logger.info(f"Copied {copied_count} images to {output_images_dir}")


@app.command()
def export_mapping_to_csv(
    sheet_path: Path = DEFAULT_MAPPING_PATH_XLSX,
    output_path: Path = DEFAULT_MAPPING_PATH_CSV,
) -> None:
    """Export a mapping worksheet from an Excel workbook to a CSV file.

    Args:
        sheet_path: Path to the Excel workbook containing the mapping sheet.
        output_path: Path where the exported CSV file should be written.

    Raises:
        FileNotFoundError: If the Excel workbook does not exist.
        ValueError: If the canonical mapping worksheet is not present in the workbook.
    """
    logger.info(f"Exporting mapping sheet '{MAPPING_SHEET_NAME}' from {sheet_path} to {output_path}")

    if not sheet_path.exists():
        raise FileNotFoundError(f"Mapping workbook not found: {sheet_path}")

    # Load workbook
    workbook = load_workbook(sheet_path, data_only=True, read_only=True)
    try:
        if MAPPING_SHEET_NAME not in workbook.sheetnames:
            raise ValueError(f"Sheet '{MAPPING_SHEET_NAME}' not found in {sheet_path}.")

        output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure outputpath exists
        worksheet = workbook[MAPPING_SHEET_NAME]

        # Write CSV file
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for row in worksheet.iter_rows(values_only=True):
                values = ["" if value is None else value for value in row]
                if any(str(value).strip() for value in values):
                    writer.writerow(values)

        logger.info(f"Exported mapping CSV to {output_path}")
    finally:
        workbook.close()


@app.command()
def preprocess(
    raw_input_dir: Path = DEFAULT_RAW_DATA_DIR,
    output_dir: Path = DEFAULT_PREPROCESS_DATA_DIR,
    mapping_path: Path = DEFAULT_MAPPING_PATH_CSV,
) -> None:
    """Remaps the labels and saves them to an output dir
    Assumes the raw structure, as it's currently present in both
    datasets"""
    logger.info(f"Preprocessing data from {raw_input_dir} into {output_dir}")

    # Initialize class mapping
    class_mapping = ClassMapping.from_csv(mapping_path)

    # Preprocess German Dataset
    for split in ["Train", "Test"]:
        input_split_dir = raw_input_dir / "GTSDB_Train_and_Test" / split
        output_split = split.lower()
        _remap_label_folder(
            input_labels_dir=input_split_dir / "labels",
            output_labels_dir=output_dir / output_split / "labels",
            class_mapping=class_mapping,
            dataset_name="germany",
        )
        _copy_image_folder(
            input_images_dir=input_split_dir / "images",
            output_images_dir=output_dir / output_split / "images",
        )

    # Preprocess Italien Dataset
    for split in ["train", "test", "valid"]:
        # Save valid split also into test dir
        input_split_dir = raw_input_dir / "StreetSignSet" / split
        output_split = "test" if split == "valid" else split
        _remap_label_folder(
            input_labels_dir=input_split_dir / "labels",
            output_labels_dir=output_dir / output_split / "labels",
            class_mapping=class_mapping,
            dataset_name="italy",
        )
        _copy_image_folder(
            input_images_dir=input_split_dir / "images",
            output_images_dir=output_dir / output_split / "images",
        )


@app.command()
def create_yaml_dataset() -> None:
    root = project_root() / "data"
    path = "data/preprocessed"
    train = "train/images"
    val = "test/images"
    mapping = ClassMapping.from_csv(DEFAULT_MAPPING_PATH_CSV)
    ids_names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(mapping.class_names, start=0))
    yaml_content = f"""path: {path}\ntrain: {train}\nval: {val}\n\nnames:\n{ids_names_yaml}
    """
    with open(str(root / "dataset.yaml"), "w") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    app()
