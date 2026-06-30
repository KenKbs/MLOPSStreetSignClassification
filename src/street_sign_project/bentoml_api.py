import os
import tempfile
from pathlib import Path
from typing import Annotated

import bentoml
import cv2
import numpy as np
from bentoml.validators import ContentType
from PIL import Image as PILImage

from street_sign_project.model import YOLOv26

DEFAULT_MODEL_NAME = "YOLO_eps420_bs8_lr0.005_fr10_x.pt"
ImageOutput = Annotated[PILImage.Image, ContentType("image/jpeg")]


@bentoml.service(name="street-sign-classifier")
class StreetSignClassifierService:
    """BentoML service for street sign object detection."""

    def __init__(self) -> None:
        """Load the YOLO model once when the BentoML service starts."""
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.model = YOLOv26(local_model_name=model_name)

    @bentoml.api(route="/image_input/")
    def image_input(self, image: PILImage.Image) -> ImageOutput:
        """Predict street signs on an uploaded image and return an annotated image."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.jpg"
            image.convert("RGB").save(input_path, format="JPEG")

            annotated_image = self._annotate_image(input_path)
            # Convert BGR color (OpenCV) back to RGB to put it back to PIL image
            response_image = PILImage.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            response_image.format = "JPEG"
            return response_image

    def _annotate_image(self, image_path: Path) -> np.ndarray:
        """Run model prediction and draw detected boxes on the input image."""
        # Read the image using OpenCV
        image = cv2.imread(str(image_path))
        # Sanity checks: image & prediction present?
        if image is None:
            raise ValueError("Image could not be loaded")

        prediction = self.model.predict(image_path)
        if prediction.boxes is None:
            raise ValueError("pred.boxes is None, but should be a list of prediction objects")

        # Draw boxes and labels on the image
        for box in prediction.boxes:
            xyxy = box.xyxy[0]
            label = f"{int(box.cls[0].item())}: {round(box.conf[0].item(), 2)}"
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.rectangle(
                image,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[0] + 80), int(xyxy[1] + 12)),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                image,
                label,
                (int(xyxy[0]), int(xyxy[1] + 12)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )

        return image
