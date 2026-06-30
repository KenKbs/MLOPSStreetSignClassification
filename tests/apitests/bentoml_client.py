import argparse
from pathlib import Path

import requests

# Default Constants which can be overwritten
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "preprocessed" / "test" / "images"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "API_uploads" / "output" / "bentoml_prediction.jpg"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def default_image_path() -> Path:
    """Return the first available test image for a local BentoML smoke request."""
    image_paths = sorted(path for path in DEFAULT_IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {DEFAULT_IMAGE_DIR}")
    return image_paths[0]


def post_image(host: str, image_path: Path, output_path: Path) -> None:
    """Post one image to the BentoML service and save the annotated response."""
    image_media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    with image_path.open("rb") as image_file:
        # Build the payload
        files = {"image": (image_path.name, image_file, image_media_type)}
        # Post request and get response
        response = requests.post(f"{host.rstrip('/')}/image_input/", files=files, timeout=60)

    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"Saved BentoML response to {output_path}")  # Save response


def main() -> None:
    """Run a local BentoML API smoke request from the command line."""
    parser = argparse.ArgumentParser(description="Send one image to the local BentoML street sign API.")
    parser.add_argument("--host", default="http://localhost:3000", help="BentoML service host.")
    parser.add_argument("--image", type=Path, help="Image to upload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for the annotated response.")
    args = parser.parse_args()

    image_path = args.image if args.image is not None else default_image_path()
    post_image(args.host, image_path, args.output)


if __name__ == "__main__":
    main()
