# Monitoring

Data-drift monitoring: extracting image features, recording production predictions, building the
reference feature set, Google Cloud Storage helpers and the Evidently drift report.

## `monitoring.image_features`

Extract tabular monitoring features (size, brightness, contrast, sharpness, colour means) from an
image.

::: street_sign_project.monitoring.image_features

## `monitoring.production_records`

Summarise predictions and write per-request production monitoring records.

::: street_sign_project.monitoring.production_records

## `monitoring.reference_features`

Generate and upload the reference image-feature set used as the drift baseline.

::: street_sign_project.monitoring.reference_features

## `monitoring.storage`

Google Cloud Storage helpers for uploading and reading monitoring data.

::: street_sign_project.monitoring.storage

## `monitoring.drift_report`

Build an Evidently data-drift report from the cloud monitoring data.

::: street_sign_project.monitoring.drift_report
