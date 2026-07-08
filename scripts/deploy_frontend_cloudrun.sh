#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="${PROJECT_ID:-streetsignproject}"
REGION="${REGION:-europe-west3}"
REPOSITORY="${REPOSITORY:-docker-registry}"
IMAGE_NAME="${IMAGE_NAME:-street-sign-frontend}"
SERVICE_NAME="${SERVICE_NAME:-street-sign-frontend}"
API_SERVICE_NAME="${API_SERVICE_NAME:-street-sign-api-finn}"
PORT="${PORT:-8080}"
CPU="${CPU:-1}"
MEMORY="${MEMORY:-1Gi}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"

REGISTRY_HOST="${REGION}-docker.pkg.dev"
REMOTE_IMAGE="${REGISTRY_HOST}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"
LATEST_IMAGE="${REGISTRY_HOST}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"
LOCAL_LATEST_IMAGE="${IMAGE_NAME}:latest"

command -v docker >/dev/null || {
  echo "docker is required but was not found." >&2
  exit 1
}

command -v gcloud >/dev/null || {
  echo "gcloud is required but was not found." >&2
  exit 1
}

command -v python3 >/dev/null || {
  echo "python3 is required but was not found." >&2
  exit 1
}

API_URL="${API_URL:-}"
if [[ -z "${API_URL}" ]]; then
  echo "Fetching API URL from Cloud Run service ${API_SERVICE_NAME}"
  service_json="$(gcloud run services describe "${API_SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format=json)"

  API_URL="$(SERVICE_JSON="${service_json}" python3 - "${REGION}" <<'PY'
import json
import os
import sys

region = sys.argv[1]
service = json.loads(os.environ["SERVICE_JSON"])
annotations = service.get("metadata", {}).get("annotations", {})
raw_urls = annotations.get("run.googleapis.com/urls", "")

try:
    urls = json.loads(raw_urls)
except json.JSONDecodeError:
    urls = []

regional_urls = [url for url in urls if f'.{region}.run.app' in url]
selected_url = regional_urls[0] if regional_urls else (urls[0] if urls else '')

if not selected_url:
    selected_url = service.get("status", {}).get("url", "")

print(selected_url)
PY
  )"
fi

if [[ -z "${API_URL}" ]]; then
  echo "Could not determine API_URL. Set API_URL explicitly or deploy API service first." >&2
  exit 1
fi

echo "Using API_URL=${API_URL}"

echo "Configuring Docker authentication for ${REGISTRY_HOST}"
gcloud auth configure-docker "${REGISTRY_HOST}" --quiet

echo "Building ${LOCAL_IMAGE}"
docker build \
  --file dockerfiles/frontend.dockerfile \
  --tag "${LOCAL_IMAGE}" \
  --tag "${LOCAL_LATEST_IMAGE}" \
  --tag "${REMOTE_IMAGE}" \
  --tag "${LATEST_IMAGE}" \
  .

echo "Pushing ${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo "Pushing ${LATEST_IMAGE}"
docker push "${LATEST_IMAGE}"

auth_args=(--no-allow-unauthenticated)
if [[ "${ALLOW_UNAUTHENTICATED}" == "true" ]]; then
  auth_args=(--allow-unauthenticated)
fi

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
  --set-env-vars "API_URL=${API_URL},STREAMLIT_LOCK_API_URL=true" \
  --min-instances=0 \
  --max-instances=1 \
  "${auth_args[@]}" \
  --quiet

echo "Routing all traffic to the latest revision"
gcloud run services update-traffic "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --to-latest \
  --quiet

echo "Deployment complete: ${REMOTE_IMAGE}"

echo "Info about running service:"
gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}"
