import random
from pathlib import Path

from locust import HttpUser, between, task

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = REPO_ROOT / "data" / "preprocessed" / "test" / "images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class StreetSignApiUser(HttpUser):
    """Locust user that stress-tests the street sign image upload endpoint."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Load the available test images once per simulated user."""
        # Load all test images
        self.image_paths = sorted(path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

        if not self.image_paths:
            raise FileNotFoundError(f"No test images found in {IMAGE_DIR}")

    @task(3)
    def upload_image(self) -> None:
        """Upload one image to the model inference endpoint."""
        image_path = random.choice(self.image_paths)  # Randomly pick one test image
        image_media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

        # Prepare Tuple to upload to API data: ("germany_test.jpg, image_bytes..., image/jpeg")
        files = {"data": (image_path.name, image_path.read_bytes(), image_media_type)}

        # Post to API
        with self.client.post("/image_input/", files=files, name="/image_input/", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Expected status 200, got {response.status_code}: {response.text}")
            elif response.headers.get("content-type") != "image/jpeg":
                response.failure(f"Expected image/jpeg response, got {response.headers.get('content-type')}")
