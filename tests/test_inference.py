"""
Inference Validation Tests
Student: Sricharan | Roll No: 2022BCS0111

Tests for validating the FastAPI prediction API:
1. Pull Docker image
2. Run container
3. Send test request
4. Verify response contains Prediction, Name, Roll No
"""

import json
import os
import sys

import pytest
import requests

# API base URL (default: localhost:8000)
API_URL = os.environ.get("API_URL", "http://localhost:8000")


# ── Sample Inputs ──────────────────────────────────────────────────────────

VALID_INPUT = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}

INVALID_INPUT = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    # Missing required fields
}


# ── Tests ──────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self):
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 200

    def test_health_contains_name(self):
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        assert "name" in data
        assert data["name"] == "Sricharan"

    def test_health_contains_roll_no(self):
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        assert "roll_no" in data
        assert data["roll_no"] == "2022BCS0111"

    def test_health_status_healthy(self):
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_valid_prediction_returns_200(self):
        response = requests.post(f"{API_URL}/predict", json=VALID_INPUT)
        assert response.status_code == 200

    def test_prediction_contains_value(self):
        response = requests.post(f"{API_URL}/predict", json=VALID_INPUT)
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))

    def test_prediction_contains_name(self):
        response = requests.post(f"{API_URL}/predict", json=VALID_INPUT)
        data = response.json()
        assert data["name"] == "Sricharan"

    def test_prediction_contains_roll_no(self):
        response = requests.post(f"{API_URL}/predict", json=VALID_INPUT)
        data = response.json()
        assert data["roll_no"] == "2022BCS0111"

    def test_prediction_in_valid_range(self):
        """Wine quality should be roughly between 0 and 10."""
        response = requests.post(f"{API_URL}/predict", json=VALID_INPUT)
        data = response.json()
        assert 0 <= data["prediction"] <= 10

    def test_invalid_input_returns_422(self):
        """Missing fields should return 422 Unprocessable Entity."""
        response = requests.post(f"{API_URL}/predict", json=INVALID_INPUT)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        """Empty body should return 422."""
        response = requests.post(f"{API_URL}/predict", json={})
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
