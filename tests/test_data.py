from pathlib import Path

import pytest
from street_sign_project.data import (
    DEFAULT_MAPPING_PATH_XLSX,
    ClassMapping,
    _validate_split_ratios,
    create_yaml_dataset,
    export_mapping_to_csv,
    preprocess,
)


def test_class_mapping_loads_from_csv(tmp_path: Path) -> None:
    """Test that ClassMapping loads class names and the dataset mappings from CSV file."""
    mapping_path = tmp_path / "mapping.csv"

    with open(mapping_path, "w", encoding="utf-8") as file:
        file.write("canonical_id,canonical_name,germany_id,italy_id\n")
        file.write("0,speed limit,1,10\n")
        file.write("1,stop,2,20\n")

    class_mapping = ClassMapping.from_csv(mapping_path)

    assert class_mapping.class_names == ("speed limit", "stop"), "class names not correctly loaded"
    assert class_mapping.germany_to_canonical == {1: 0, 2: 1}, "germany_to_canonical mapping not correctly loaded"
    assert class_mapping.italy_to_canonical == {10: 0, 20: 1}, "italy_to_canonical mapping not correctly loaded"


@pytest.mark.skipif(not DEFAULT_MAPPING_PATH_XLSX.exists(), reason="Mapping workbook not found")
def test_export_mapping_to_csv_exports_real_workbook(tmp_path: Path) -> None:
    """Test that the real mapping workbook can be exported to a usable CSV file."""
    output_path = tmp_path / "street_sign_class_mapping.csv"

    export_mapping_to_csv(
        sheet_path=DEFAULT_MAPPING_PATH_XLSX,
        output_path=output_path,
    )

    class_mapping = ClassMapping.from_csv(output_path)

    assert output_path.exists(), "Mapping CSV was not created"
    assert len(class_mapping.class_names) > 0, "Exported mapping CSV did not contain class names"
    assert len(class_mapping.italy_to_canonical) > 0, "Exported mapping CSV did not contain Italy mappings"
    assert len(class_mapping.germany_to_canonical) > 0, "Exported mapping CSV did not contain Germany mappings"


def _write_yolo_item(split_dir: Path, image_name: str, class_id: int) -> None:
    """Write one fake YOLO image and matching label file."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / image_name
    image_path.write_bytes(b"fake image bytes")
    with open(labels_dir / f"{image_path.stem}.txt", "w", encoding="utf-8") as file:
        file.write(f"{class_id} 0.5 0.5 0.2 0.2\n")


def test_preprocess_runs_on_mini_raw_dataset(tmp_path: Path) -> None:
    """Test the preprocessing pipeline on a tiny fake raw dataset.
    1.) Creates five fake fake YOLO image and five fake label file into all "raw" directories
        (train, test Germany and train,test,val for Italy) (ARRANGE)
    2.) Writes a small csv mapping file (independent from other test) (ARRANGE)
    3.) then it calls the real preprocess function (ACT)
    4.) asserts that preprocessing worked (Assert)
    Note: everything is beeing created in tmp_path so real data is not affectd.
    """
    # Define directories
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "preprocessed"
    mapping_path = tmp_path / "mapping.csv"

    # Write mapping file
    with open(mapping_path, "w", encoding="utf-8") as file:
        file.write("canonical_id,canonical_name,germany_id,italy_id\n")
        file.write("0,speed limit,1,10\n")
        file.write("1,stop,2,20\n")

    # Write fake Yolo images and labels
    _write_yolo_item(raw_dir / "GTSDB_Train_and_Test" / "Train", "germany_train.jpg", 1)
    _write_yolo_item(raw_dir / "GTSDB_Train_and_Test" / "Test", "germany_test.jpg", 2)
    _write_yolo_item(raw_dir / "StreetSignSet" / "train", "italy_train.jpg", 10)
    _write_yolo_item(raw_dir / "StreetSignSet" / "test", "italy_test.jpg", 20)
    _write_yolo_item(raw_dir / "StreetSignSet" / "valid", "italy_valid.jpg", 10)

    # ACT call preprocess function
    preprocess(
        raw_input_dir=raw_dir,
        output_dir=output_dir,
        mapping_path=mapping_path,
        train_ratio=0.6,
        valid_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    # Get all image and label output paths
    split_image_paths = list(output_dir.glob("*/images/*.jpg"))
    split_label_paths = list(output_dir.glob("*/labels/*.txt"))

    # Combine into one big string
    label_text = "\n".join(label_path.read_text(encoding="utf-8") for label_path in split_label_paths)

    # Assert checks!
    assert len(split_image_paths) == 5, "Data split returned wrong number of images (5 expected for mock data)"
    assert len(split_label_paths) == 5, "Data split returned wrong number of labels (5 expected for mock data)"
    assert not (output_dir / "_split_pool").exists(), "Temporary split pool directory was not cleaned up"
    assert {path.parent.parent.name for path in split_image_paths} == {
        "train",
        "valid",
        "test",
    }, "Data split returned wrong parent directories (train, valid, test expected)"
    assert "10 " not in label_text, "Original Italy class ID 10 was not remapped"
    assert "20 " not in label_text, "Original Italy class ID 20 was not remapped"
    assert "0 0.5 0.5 0.2 0.2" in label_text, "Canonical class ID 0 was missing from remapped labels"
    assert "1 0.5 0.5 0.2 0.2" in label_text, "Canonical class ID 1 was missing from remapped labels"


def test_create_yaml_dataset_writes_expected_yaml(tmp_path: Path) -> None:
    """Test that create_yaml_dataset writes a YOLO dataset YAML file."""
    mapping_path = tmp_path / "mapping.csv"
    output_path = tmp_path / "dataset.yaml"

    # Again same structure: Arrange
    with open(mapping_path, "w", encoding="utf-8") as file:
        file.write("canonical_id,canonical_name,germany_id,italy_id\n")
        file.write("0,speed limit,1,10\n")
        file.write("1,stop,2,20\n")

    # Act
    create_yaml_dataset(
        output_path=output_path,
        mapping_path=mapping_path,
    )

    # Assert:
    assert output_path.read_text(encoding="utf-8") == (
        "path: data/preprocessed\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "names:\n"
        "  0: speed limit\n"
        "  1: stop\n"
    ), "The dataset.yaml file was not created correctly."


def test_validate_split_ratios() -> None:
    """Test that invalid split ratios raise a ValueError."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        _validate_split_ratios(train_ratio=0.7, valid_ratio=0.2, test_ratio=0.2)
