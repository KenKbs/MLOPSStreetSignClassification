# Model & training

The YOLO model wrapper and the Hydra-driven training orchestrator.

## `model`

The `YOLOv26` wrapper class around Ultralytics YOLO (init from a pretrained or local model, train,
predict, validate with Ultralytics through `test`, and save checkpoints). Training logs metrics and the saved model artifact to W&B.

::: street_sign_project.model

## `train`

The Hydra entry point that maps the configuration onto `YOLOv26.train`.

::: street_sign_project.train
