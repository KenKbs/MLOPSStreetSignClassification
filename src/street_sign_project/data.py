from pathlib import Path

import typer
from torch.utils.data import Dataset


# Define Path constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_GERMAN_DATA_DIR = DEFAULT_RAW_DATA_DIR / "Germany-GTSDB_Train_and_Test"
DEFAULT_ITALIEN_DATA_DIR = DEFAULT_RAW_DATA_DIR / "Italy-StreetSignSet"

DEFAULT_PREPROCESSED_DATA_DIR = PROJECT_ROOT / "preprocessed"
DEFAULT_MAPPING_XLSX = DEFAULT_PREPROCESSED_DATA_DIR / "street_sign_class_mapping.xlsx"
DEFAULT_MAPPING_CSV = DEFAULT_PREPROCESSED_DATA_DIR / "street_sign_class_mapping.csv"

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

def export_mapping_to_csv(sheet_path: Path, 
                          output_Path:Path) -> None:
    pass

def load_class_mapping(mapping_path:Path) -> None:
    pass

def remap_label_file(data_path:Path) -> None:
    pass


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
