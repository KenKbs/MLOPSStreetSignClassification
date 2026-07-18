# Model & training

The YOLO model wrapper, the Hydra-driven training orchestrator and the evaluation metrics.

## `model`

The `YOLOv26` wrapper class around Ultralytics YOLO (init from a pretrained or local model, train,
predict, test, save, evaluate and model-quality bookkeeping).

::: street_sign_project.model

## `train`

The Hydra entry point that maps the configuration onto `YOLOv26.train`.

::: street_sign_project.train

## `evaluate`

True-positive / false-positive / false-negative counting and the models-quality YAML builder.

::: street_sign_project.evaluate
