"""Streamlit frontend for uploading images to the FastAPI prediction endpoint."""
# pyright: reportMissingImports=false

from __future__ import annotations

import os
import queue
import threading
import time
from io import BytesIO
from typing import Any

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


def _ensure_prediction_state() -> None:
    """Initialize session-state entries used by the prediction workflow."""
    defaults: dict[str, Any] = {
        "prediction_running": False,
        "prediction_cancel_requested": False,
        "prediction_cancel_event": None,
        "prediction_queue": None,
        "prediction_result": None,
        "prediction_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _prediction_worker(
    api_base_url: str,
    image_bytes: bytes,
    filename: str,
    timeout_seconds: float,
    result_queue: queue.Queue[tuple[str, bytes | str]],
    cancel_event: threading.Event,
) -> None:
    """Execute prediction in a worker thread and push result into a queue."""
    if cancel_event.is_set():
        return
    try:
        result = _predict_image(
            api_base_url=api_base_url,
            image_bytes=image_bytes,
            filename=filename,
            timeout_seconds=timeout_seconds,
        )
    except requests.HTTPError as error:
        response = error.response
        if response is None:
            result_queue.put(("error", f"API returned an error: {error}"))
        else:
            result_queue.put(("error", f"API returned an error: {response.status_code} {response.text}"))
        return
    except requests.RequestException as error:
        result_queue.put(("error", f"Could not reach API at {api_base_url}: {error}"))
        return

    if not cancel_event.is_set():
        result_queue.put(("success", result))


def _poll_prediction_result() -> None:
    """Move finished worker results from queue into session state."""
    result_queue = st.session_state.get("prediction_queue")
    if result_queue is None:
        return
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        return

    st.session_state["prediction_running"] = False
    st.session_state["prediction_queue"] = None
    st.session_state["prediction_cancel_event"] = None

    if st.session_state.get("prediction_cancel_requested", False):
        st.session_state["prediction_cancel_requested"] = False
        st.session_state["prediction_result"] = None
        st.session_state["prediction_error"] = "Prediction stopped."
        return

    if status == "success":
        st.session_state["prediction_result"] = payload
        st.session_state["prediction_error"] = None
    else:
        st.session_state["prediction_result"] = None
        st.session_state["prediction_error"] = str(payload)


def main() -> None:
    """Render the Streamlit frontend for image upload and prediction."""
    st.set_page_config(page_title="Street Sign Predictor", page_icon="🚦", layout="wide")
    st.markdown(
        """
        <style>
        .predict-inline-loader {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-top: 0.5rem;
            color: #4b5563;
            font-size: 0.95rem;
            font-weight: 500;
        }
        .predict-spinner {
            width: 1rem;
            height: 1rem;
            border: 0.15rem solid #cfd8e3;
            border-top-color: #1d4ed8;
            border-radius: 999px;
            animation: predict-spin 0.8s linear infinite;
        }
        @keyframes predict-spin {
            100% { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Street Sign Prediction Frontend")
    _ensure_prediction_state()

    st.sidebar.header("API Settings")
    default_api_url = _api_base_url()
    if _api_url_locked():
        st.sidebar.caption("API URL is managed by the deployment environment.")
        api_base_url = st.sidebar.text_input("FastAPI URL", value=default_api_url, disabled=True)
    else:
        api_base_url = st.sidebar.text_input("FastAPI URL", value=default_api_url)
    timeout_seconds = st.sidebar.number_input("Request timeout (seconds)", min_value=1.0, max_value=120.0, value=30.0)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image to run prediction.")
        return

    file_bytes = uploaded_file.getvalue()
    st.subheader("Input")
    st.image(file_bytes, caption=uploaded_file.name, use_container_width=True)

    _poll_prediction_result()

    predict_button_placeholder = st.empty()
    loader_placeholder = st.empty()

    if st.session_state["prediction_running"]:
        if predict_button_placeholder.button("Stop", type="secondary", key="stop_prediction"):
            cancel_event = st.session_state.get("prediction_cancel_event")
            if cancel_event is not None:
                cancel_event.set()
            st.session_state["prediction_running"] = False
            st.session_state["prediction_cancel_requested"] = True
            st.session_state["prediction_result"] = None
            st.session_state["prediction_error"] = "Prediction stopped."
            loader_placeholder.empty()
            st.rerun()

        loader_placeholder.markdown(
            """
            <div class="predict-inline-loader">
                <span class="predict-spinner"></span>
                <span>Predicting...</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if predict_button_placeholder.button("Predict", type="primary", key="run_prediction"):
            result_queue: queue.Queue[tuple[str, bytes | str]] = queue.Queue()
            cancel_event = threading.Event()
            worker = threading.Thread(
                target=_prediction_worker,
                kwargs={
                    "api_base_url": api_base_url,
                    "image_bytes": file_bytes,
                    "filename": uploaded_file.name,
                    "timeout_seconds": float(timeout_seconds),
                    "result_queue": result_queue,
                    "cancel_event": cancel_event,
                },
                daemon=True,
            )
            st.session_state["prediction_running"] = True
            st.session_state["prediction_cancel_requested"] = False
            st.session_state["prediction_cancel_event"] = cancel_event
            st.session_state["prediction_queue"] = result_queue
            st.session_state["prediction_result"] = None
            st.session_state["prediction_error"] = None
            worker.start()
            st.rerun()

    prediction_error = st.session_state.get("prediction_error")
    if prediction_error:
        st.warning(prediction_error)

    result_image = st.session_state.get("prediction_result")
    if result_image is not None:
        st.subheader("Prediction Result")
        st.image(result_image, caption="Annotated image", use_container_width=True)
        st.download_button(
            label="Download result image",
            data=BytesIO(result_image),
            file_name=f"pred_{uploaded_file.name}",
            mime="image/jpeg",
        )

    if st.session_state["prediction_running"]:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
