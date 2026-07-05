#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="${PROJECT_ID:-mlops-steetsigns}"
REGION="${REGION:-europe-west3}"
REPOSITORY="${REPOSITORY:-docker-registry}"
IMAGE_NAME="${IMAGE_NAME:-street-sign-api}"
SERVICE_NAME="${SERVICE_NAME:-street-sign-api}"
MODEL_NAME="${MODEL_NAME:-YOLO_eps420_bs8_lr0.005_fr10_x.pt}"
MONITORING_BUCKET="${MONITORING_BUCKET:-mlops-street-signs-prod-data}"
MONITORING_PREFIX="${MONITORING_PREFIX:-production}"
PORT="${PORT:-8000}"
CPU="${CPU:-1}"
MEMORY="${MEMORY:-2Gi}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"

MODEL_PATH="models/${MODEL_NAME}"
REGISTRY_HOST="${REGION}-docker.pkg.dev"
REMOTE_IMAGE="${REGISTRY_HOST}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"
LATEST_IMAGE="${REGISTRY_HOST}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"
LOCAL_LATEST_IMAGE="${IMAGE_NAME}:latest"

# Sanity checks
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model file not found: ${MODEL_PATH}" >&2
  echo "Pull it first, for example: uv run dvc pull ${MODEL_PATH}.dvc" >&2
  exit 1
fi

command -v docker >/dev/null || {
  echo "docker is required but was not found." >&2
  exit 1
}

command -v gcloud >/dev/null || {
  echo "gcloud is required but was not found." >&2
  exit 1
}

# Allow local docker to use GC
echo "Configuring Docker authentication for ${REGISTRY_HOST}"
gcloud auth configure-docker "${REGISTRY_HOST}" --quiet

# Build image
echo "Building ${LOCAL_IMAGE} with ${MODEL_PATH}"
docker build \
  --file dockerfiles/api.dockerfile \
  --build-arg "API_MODEL_NAME=${MODEL_NAME}" \
  --tag "${LOCAL_IMAGE}" \
  --tag "${LOCAL_LATEST_IMAGE}" \
  --tag "${REMOTE_IMAGE}" \
  --tag "${LATEST_IMAGE}" \
  .

# Push image
echo "Pushing ${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo "Pushing ${LATEST_IMAGE}"
docker push "${LATEST_IMAGE}"

auth_args=(--no-allow-unauthenticated)
if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
  auth_args=(--allow-unauthenticated)
fi

# Deploy image
echo "Deploying ${SERVICE_NAME} to Cloud Run in ${REGION}"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${REMOTE_IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --port "${PORT}" \
  --cpu "${CPU}" \
  --memory "${MEMORY}" \
  --cpu-throttling \
  --set-env-vars "MODEL_NAME=${MODEL_NAME},MONITORING_BUCKET=${MONITORING_BUCKET},MONITORING_PREFIX=${MONITORING_PREFIX}" \
  --min-instances=0 \
  --max-instances=1 \
  "${auth_args[@]}" \
  --quiet

# Deploymnet alone not enough, need to route traffic to latest version
echo "Routing all traffic to the latest revision"
gcloud run services update-traffic "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --to-latest \
  --quiet

echo "Deployment complete: ${REMOTE_IMAGE}"
