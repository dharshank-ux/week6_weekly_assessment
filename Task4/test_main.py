import pytest
from fastapi.testclient import TestClient
import main

API_KEY = "week6-local-secret-key"

headers = {
    "X-API-Key": API_KEY
}


@pytest.fixture(scope="module")
def client():
    # This context manager runs FastAPI startup and shutdown
    with TestClient(main.app) as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_search_without_api_key(client):
    response = client.post(
        "/search",
        json={
            "query": "employee working hours",
            "top_k": 3
        }
    )
    assert response.status_code == 422


def test_search_with_invalid_api_key(client):
    response = client.post(
        "/search",
        headers={"X-API-Key": "wrong-key"},
        json={
            "query": "employee working hours",
            "top_k": 3
        }
    )
    assert response.status_code == 401


def test_search_with_valid_api_key(client):
    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": "employee working hours",
            "top_k": 3
        }
    )

    assert response.status_code == 200
    assert "results" in response.json()
    assert response.json()["top_k"] == 3


def test_search_invalid_top_k(client):
    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": "employee working hours",
            "top_k": 25
        }
    )
    assert response.status_code == 422