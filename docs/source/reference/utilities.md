# Utilities

Shared helper utilities and dataset-annotation drawing.

## `utils`

Small shared helpers (e.g. locating the project root).

::: street_sign_project.utils

## `visualize`

The `plot_image_label` helper draws bounding boxes from dataset annotations and saves or displays the images.
Dataset statistics call it directly to save annotated examples. This module has no command-line interface;
prediction images returned by FastAPI and BentoML are drawn in their respective serving modules.

::: street_sign_project.visualize
