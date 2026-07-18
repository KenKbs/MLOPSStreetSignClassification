# Data pipeline

Loading, remapping and splitting the raw traffic-sign datasets into a YOLO-ready dataset, plus the
dataset-statistics utilities.

## `data`

Class mapping, label remapping and the deterministic, rarity-aware train/valid/test split.

::: street_sign_project.data

## `dataset`

PyTorch dataset wrapper and dataset-statistics command used by the data-checker CI workflow.

::: street_sign_project.dataset
