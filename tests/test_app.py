"""
Basic smoke tests for the Rising Waters Flask application.

Run:
    python -m pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        yield client


def test_index_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Rising Waters" in response.data


def test_predict_route_handles_post(client):
    payload = {
        "winter_rainfall": "30",
        "pre_monsoon_rainfall": "260",
        "monsoon_rainfall": "2400",
        "post_monsoon_rainfall": "480",
    }
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
