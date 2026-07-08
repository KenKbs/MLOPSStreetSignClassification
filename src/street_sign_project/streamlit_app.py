"""Streamlit frontend for uploading images to the FastAPI prediction endpoint."""

from __future__ import annotations

import os
from io import BytesIO

import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = "/image_input/"


def _api_url_locked() -> bool:
    """Return whether the API URL should be fixed from the environment."""
    return os.getenv("STREAMLIT_LOCK_API_URL", "false").lower() == "true"


def _api_base_url() -> str:
    """Return the configured API base URL without trailing slash."""
    return os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")


def _predict_image(api_base_url: str, image_bytes: bytes, filename: str, timeout_seconds: float) -> bytes:
    """Send one image to the API and return the annotated image bytes."""
    url = f"{api_base_url}{PREDICT_ENDPOINT}"
    files = {"data": (filename, image_bytes, "image/jpeg")}
    response = requests.post(url, files=files, timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


def main() -> None:
    """Render the Streamlit frontend for image upload and prediction."""
    st.set_page_config(page_title="Street Sign Predictor", page_icon="🚦", layout="wide")
    st.title("Street Sign Prediction Frontend")

    st.sidebar.header("API Settings")
    default_api_url = _api_base_url()
    if _api_url_locked():
        st.sidebar.caption("API URL is managed by the deployment environment.")
        api_base_url = st.sidebar.text_input("FastAPI URL", value=default_api_url, disabled=True)
    else:
        api_base_url = st.sidebar.text_input(
            "FastAPI URL", value=default_api_url, help="Example: http://localhost:8000"
        )
    timeout_seconds = st.sidebar.number_input("Request timeout (seconds)", min_value=1.0, max_value=120.0, value=30.0)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image to run prediction.")
        return

    file_bytes = uploaded_file.getvalue()
    st.subheader("Input")
    st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)

    if st.button("Run Prediction", type="primary"):
        try:
            result_image = _predict_image(
                api_base_url=api_base_url,
                image_bytes=file_bytes,
                filename=uploaded_file.name,
                timeout_seconds=float(timeout_seconds),
            )
        except requests.HTTPError as error:
            response = error.response
            if response is None:
                st.error(f"API returned an error: {error}")
            else:
                st.error(f"API returned an error: {response.status_code} {response.text}")
            return
        except requests.RequestException as error:
            st.error(f"Could not reach API at {api_base_url}: {error}")
            return

        st.subheader("Prediction Result")
        st.image(result_image, caption="Annotated image", use_container_width=True)
        st.download_button(
            label="Download result image",
            data=BytesIO(result_image),
            file_name=f"pred_{uploaded_file.name}",
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
