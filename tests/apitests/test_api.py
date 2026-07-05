from fastapi.testclient import TestClient
from street_sign_project.fast_api import app

client = TestClient(app)


def test_openapi_schema_includes_image_input_route() -> None:
    """Test that the API exposes the image upload route."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/image_input/" in response.json()["paths"]
