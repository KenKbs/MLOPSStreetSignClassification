# Street Sign Project

The project serves a YOLO street-sign detector through FastAPI and deploys it to Google Cloud Run.

## API deployment

The `Test, build, and deploy FastAPI` GitHub Actions workflow is the primary production deployment path. On relevant
pushes to `main`, or when started manually, it:

1. installs the locked development environment and runs the test suite;
2. downloads the configured API model from the DVC GCS remote;
3. submits the FastAPI container build to Google Cloud Build;
4. pushes commit-specific and `latest` image tags to Artifact Registry; and
5. deploys the commit-specific image to Cloud Run.

Configure the service-account JSON as the `GCP_API_DEPLOY_CREDENTIALS` repository secret. Grant that service account
`roles/storage.objectViewer` on the DVC bucket, `roles/storage.objectUser` on the Cloud Build staging bucket,
`roles/cloudbuild.builds.editor` and `roles/run.admin` on the project, `roles/artifactregistry.reader` on the image
repository, and `roles/iam.serviceAccountUser` on the Cloud Run runtime service account. The Cloud Build service account
needs `roles/artifactregistry.writer` on the image repository. The Cloud Run runtime service account needs
`roles/storage.objectUser` on the monitoring bucket.

The workflow supports these optional repository variables:

| Variable | Default |
| --- | --- |
| `GCP_PROJECT_ID` | `mlops-steetsigns` |
| `GCP_REGION` | `europe-west3` |
| `GAR_REPOSITORY` | `docker-registry` |
| `API_IMAGE_NAME` | `street-sign-api` |
| `CLOUD_RUN_API_SERVICE` | `street-sign-api` |
| `API_MODEL_NAME` | `YOLO_eps420_bs8_lr0.005_fr10_x.pt` |
| `MONITORING_BUCKET` | `mlops-street-signs-prod-data` |
| `MONITORING_PREFIX` | `production` |
| `API_PORT` | `8000` |
| `CLOUD_RUN_CPU` | `2` |
| `CLOUD_RUN_MEMORY` | `4Gi` |
| `CLOUD_RUN_MIN_INSTANCES` | `0` |
| `CLOUD_RUN_MAX_INSTANCES` | `1` |
| `ALLOW_UNAUTHENTICATED` | `true` |
| `CLOUD_BUILD_STAGING_BUCKET` | `mlops-steetsigns_cloudbuild` |

For local fallback deployment, use:

```bash
uv run invoke deploy-api
```
