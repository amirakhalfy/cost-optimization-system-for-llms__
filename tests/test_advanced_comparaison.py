import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app
import pytest
from fastapi.testclient import TestClient
from typing import List


@pytest.fixture(scope="module")
def test_client() -> TestClient:
    """
    Fixture providing a FastAPI TestClient instance for the entire test module.
    """
    client = TestClient(app)
    yield client


@pytest.fixture
def valid_benchmark_name(test_client: TestClient) -> str:
    """
    Fixture returning a valid benchmark name from the database.
    """
    response = test_client.get("/advancedaggregation/best-value-models", params={"benchmark_name": "MMLU", "page_size": 1})
    if response.status_code == 200:
        return "MMLU"
    pytest.skip("No valid benchmark available for tests")


@pytest.fixture
def invalid_benchmark_name() -> str:
    """
    Fixture returning an invalid benchmark name.
    """
    return "non-existent-benchmark-xyz"


@pytest.fixture
def invalid_task_name() -> str:
    """
    Fixture returning an invalid task name.
    """
    return "non-existent-task-xyz"


class TestBestValueModels:

    def test_valid_benchmark_basic(self, test_client: TestClient, valid_benchmark_name: str):
        """
        Test that the endpoint returns valid results for a known benchmark.
        """
        response = test_client.get(
            "/advancedaggregation/best-value-models",
            params={"benchmark_name": valid_benchmark_name, "page": 1, "page_size": 5, "min_score": 1.0}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            item = data[0]
            assert "model_id" in item and isinstance(item["model_id"], int)
            assert "model_name" in item and isinstance(item["model_name"], str)
            assert "provider_name" in item and isinstance(item["provider_name"], str)
            assert "benchmark_name" in item and isinstance(item["benchmark_name"], str)
            assert "score" in item and isinstance(item["score"], float)
            assert "cost_per_point" in item and isinstance(item["cost_per_point"], float)
            assert "cost_efficiency_rank" in item and isinstance(item["cost_efficiency_rank"], int)

    def test_invalid_benchmark(self, test_client: TestClient, invalid_benchmark_name: str):
        """
        Test that invalid benchmark returns 422 error with proper message.
        """
        response = test_client.get(
            "/advancedaggregation/best-value-models",
            params={"benchmark_name": invalid_benchmark_name}
        )
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_no_models_found(self, test_client: TestClient, valid_benchmark_name: str):
        """
        Test that no models found returns 404 or 200 with empty list.

        Use a high but valid min_score (<= 100) to try forcing no result.
        """
        response = test_client.get(
            "/advancedaggregation/best-value-models",
            params={"benchmark_name": valid_benchmark_name, "min_score": 99.9, "page": 1, "page_size": 5}
        )
        if response.status_code == 404:
            assert "detail" in response.json()
        else:
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_pagination(self, test_client: TestClient, valid_benchmark_name: str):
        """
        Test pagination by comparing results of page 1 and page 2.
        """
        resp1 = test_client.get(
            "/advancedaggregation/best-value-models",
            params={"benchmark_name": valid_benchmark_name, "page": 1, "page_size": 1}
        )
        resp2 = test_client.get(
            "/advancedaggregation/best-value-models",
            params={"benchmark_name": valid_benchmark_name, "page": 2, "page_size": 1}
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        data1 = resp1.json()
        data2 = resp2.json()
        if data1 and data2:
            assert data1[0]["model_id"] != data2[0]["model_id"]


class TestOpenSourceModels:

    def test_open_source_no_task_filter(self, test_client: TestClient):
        """
        Test that the endpoint returns open-source models without any task filter.
        """
        response = test_client.get("/advancedaggregation/open-source-models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        for model in data:
            assert "model_id" in model and isinstance(model["model_id"], int)
            assert "model_name" in model and isinstance(model["model_name"], str)
            assert "provider_name" in model and isinstance(model["provider_name"], str)
            assert "tasks" in model and isinstance(model["tasks"], list)

    def test_open_source_with_invalid_task(self, test_client: TestClient, invalid_task_name: str):
        """
        Test filtering by invalid task returns an empty list.
        """
        response = test_client.get("/advancedaggregation/open-source-models", params={"task_name": invalid_task_name})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
