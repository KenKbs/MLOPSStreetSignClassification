# Serving & APIs

The FastAPI application, the specialised BentoML service and the Streamlit frontend.

## `fast_api`

FastAPI application exposing image prediction and the Evidently monitoring report.

::: street_sign_project.fast_api

## `bentoml_api`

Specialised BentoML service for street-sign object detection.

::: street_sign_project.bentoml_api

## `streamlit_app`

Streamlit frontend that uploads images to the FastAPI prediction endpoint and shows the result.

::: street_sign_project.streamlit_app
