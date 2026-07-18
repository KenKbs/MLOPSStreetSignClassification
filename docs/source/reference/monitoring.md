# Monitoring

Data-drift monitoring: extracting image features, recording production predictions, building the
reference feature set, Google Cloud Storage helpers and the Evidently drift report.

## System metrics

The FastAPI service exposes a custom Prometheus registry at `GET /metrics`. It contains the prediction request count,
prediction error count, and prediction latency histogram. Start the API locally, send image requests, and inspect the
metrics at <http://localhost:8000/metrics>.

These Prometheus metrics are independent of Google Cloud Monitoring and are not sent to Google Cloud.

## Google Cloud monitoring and alerting

Cloud Run already reports request count, request latency, response codes, CPU usage, and memory usage in the service's
**Metrics** tab. Invoke the deployed service a few times to confirm that these built-in charts update.

Create the course alert manually in **Monitoring > Alerting > Create policy**:

1. Select the **Cloud Run Revision** resource and **Request count** metric.
2. Filter `service_name` to `street-sign-api` and `response_code_class` to `5xx`.
3. Set the condition to trigger when the summed request count is greater than zero over five minutes.
4. Add an email notification channel and save the policy as `Street Sign API 5xx responses`.
5. Upload a non-image file to the prediction endpoint, then verify the 500 response, incident, and email notification.

This alert uses Cloud Run's built-in metrics and does not depend on the `/metrics` endpoint.

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
