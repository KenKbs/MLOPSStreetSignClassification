import csv
from dataclasses import dataclass
from pathlib import Path

import typer
from openpyxl import load_workbook
from torch.utils.data import Dataset


# Define Path constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_GERMAN_DATA_DIR = DEFAULT_RAW_DATA_DIR / "Germany-GTSDB_Train_and_Test"
DEFAULT_ITALIEN_DATA_DIR = DEFAULT_RAW_DATA_DIR / "Italy-StreetSignSet"

DEFAULT_PREPROCESS_DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
DEFAULT_MAPPING_PATH_XLSX = DEFAULT_PREPROCESS_DATA_DIR / "street_sign_class_mapping.xlsx"
DEFAULT_MAPPING_PATH_CSV = DEFAULT_PREPROCESS_DATA_DIR / "street_sign_class_mapping.csv"
MAPPING_SHEET_NAME = "canonical_mapping"


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

        return cls(
            class_names=tuple(class_names),
            germany_to_canonical=germany_to_canonical,
            italy_to_canonical=italy_to_canonical,
        )


class StreetSignDataset(Dataset):
    """Dataset for loading already-processed street sign data"""
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


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
    if not sheet_path.exists():
        raise FileNotFoundError(f"Mapping workbook not found: {sheet_path}")

    # Load workbook
    workbook = load_workbook(sheet_path, data_only=True, read_only=True)
    try:
        if MAPPING_SHEET_NAME not in workbook.sheetnames:
            raise ValueError(f"Sheet '{MAPPING_SHEET_NAME}' not found in {sheet_path}.")

        output_path.parent.mkdir(parents=True, exist_ok=True) # ensure outputpath exists
        worksheet = workbook[MAPPING_SHEET_NAME]

        # Write CSV file
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for row in worksheet.iter_rows(values_only=True):
                values = ["" if value is None else value for value in row]
                if any(str(value).strip() for value in values):
                    writer.writerow(values)
    finally:
        workbook.close()



def remap_label_file(data_path:Path) -> None:
    pass


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
