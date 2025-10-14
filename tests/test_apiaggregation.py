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
def valid_model_names() -> List[str]:
    """
    Fixture returning a list of valid model names.
    """
    return ["gpt-4", "claude-3-opus", "gpt-3.5-turbo"]


@pytest.fixture
def valid_provider_names() -> List[str]:
    """
    Fixture returning a list of valid provider names.
    """
    return ["OpenAI", "Anthropic", "Google"]


@pytest.fixture
def invalid_model_name() -> str:
    """
    Fixture returning an invalid model name.
    """
    return "non-existent-model-xyz-123"


@pytest.fixture
def invalid_provider_name() -> str:
    """
    Fixture returning an invalid provider name.
    """
    return "NonExistentProvider123"


def validate_model_basic_structure(model: dict) -> None:
    """
    Assert that the basic required fields exist in a model dictionary
    and are of expected types.
    Accepts 'id' as str or int (to reflect API responses).
    """
    required_fields = ["id", "name"]
    for field in required_fields:
        assert field in model, f"Missing required field: {field}"
        assert model[field] is not None, f"Field {field} cannot be None"
        if field == "id":
            assert isinstance(model[field], (str, int)), f"Field {field} must be str or int"
        else:
            assert isinstance(model[field], str), f"Field {field} must be a string"

    # Allow tasks_covered to be missing or None when not requested
    if "tasks_covered" in model and model["tasks_covered"] is not None:
        assert isinstance(model["tasks_covered"], list), "tasks_covered must be a list if present"


def validate_model_with_tasks(model: dict) -> None:
    """
    Validate model dictionary including tasks_covered as a list.
    """
    validate_model_basic_structure(model)
    assert "tasks_covered" in model, "Missing tasks_covered field"
    assert isinstance(model["tasks_covered"], list), "tasks_covered must be a list"


def validate_comparison_model(model: dict) -> None:
    """
    Validate the structure and types of a model comparison response item.
    Accepts model_id as str or int.
    """
    required_fields = {
        "model_id": (str, int),
        "model_name": str,
        "provider_name": str,
        "input_cost": (int, float),
        "output_cost": (int, float),
        "cached_input": (int, float, type(None)),
        "token_unit": str,
        "currency": str,
        "tasks": list,
        "context_window": (int, type(None)),
        "parameters": (str, int, type(None))
    }
    for field, expected_type in required_fields.items():
        assert field in model, f"Missing required field: {field}"
        if model[field] is not None:
            assert isinstance(model[field], expected_type), \
                f"Field {field} must be {expected_type}, got {type(model[field])}"

    if model["input_cost"] is not None:
        assert model["input_cost"] >= 0, "input_cost must be non-negative"
    if model["output_cost"] is not None:
        assert model["output_cost"] >= 0, "output_cost must be non-negative"


def validate_provider_structure(provider: dict) -> None:
    """
    Validate the structure and types of a provider comparison response item.
    Accepts provider_id as optional and as str or int.
    """
    required_fields = {
        # 'provider_id' is optional due to API response absence
        "provider_name": str,
        "model_count": int,
        "avg_input_cost": (int, float, type(None)),
        "avg_output_cost": (int, float, type(None)),
        "benchmark_performance": dict,
        "tasks_covered": list
    }

    for field, expected_type in required_fields.items():
        assert field in provider, f"Missing required field: {field}"
        if provider[field] is not None:
            assert isinstance(provider[field], expected_type), \
                f"Field {field} must be {expected_type}, got {type(provider[field])}"

    # provider_id if present
    if "provider_id" in provider:
        assert isinstance(provider["provider_id"], (str, int)), \
            f"Field provider_id must be str or int, got {type(provider['provider_id'])}"

    assert provider["model_count"] >= 0, "model_count must be non-negative"
    if provider["avg_input_cost"] is not None:
        assert provider["avg_input_cost"] >= 0, "avg_input_cost must be non-negative"
    if provider["avg_output_cost"] is not None:
        assert provider["avg_output_cost"] >= 0, "avg_output_cost must be non-negative"


class TestAvailableModels:

    def test_get_available_models_basic(self, test_client: TestClient):
        """
        Test the /aggregation/available-models endpoint returns a list
        of models without mandatory tasks_covered field by default.
        Accepts tasks_covered=None as valid.
        """
        response = test_client.get("/aggregation/available-models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        if models:
            for model in models:
                validate_model_basic_structure(model)
                # allow 'tasks_covered' if None or empty list when not requested
                if "tasks_covered" in model:
                    assert model["tasks_covered"] in (None, []), \
                        "tasks_covered should be None or empty if present when not requested"

    def test_get_available_models_with_tasks(self, test_client: TestClient):
        """
        Test the /aggregation/available-models endpoint returns models with tasks_covered list.
        """
        response = test_client.get("/aggregation/available-models?include_tasks=true")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        if models:
            for model in models:
                validate_model_with_tasks(model)


class TestCompareModels:

    def test_compare_models_valid_names(self, test_client: TestClient, valid_model_names: List[str]):
        """
        Test comparison of valid model names returns correctly structured data.
        """
        test_models = valid_model_names[:2]
        response = test_client.get("/aggregation/compare-models", params={"model_names": test_models})
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            for model in data:
                validate_comparison_model(model)

    def test_compare_models_invalid_names(self, test_client: TestClient, invalid_model_name: str):
        """
        Test comparison with invalid model name returns 404 with error details.
        """
        response = test_client.get("/aggregation/compare-models", params={"model_names": [invalid_model_name]})
        assert response.status_code == 404
        error_response = response.json()
        assert "detail" in error_response or "message" in error_response


class TestProviderComparison:

    def test_provider_comparison_success(self, test_client: TestClient):
        """
        Test /aggregation/provider-comparison returns a list of providers with valid structure.
        """
        response = test_client.get("/aggregation/provider-comparison")
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        if result:
            for provider in result:
                validate_provider_structure(provider)


class TestProviderInfo:

    def test_provider_info_valid_name(self, test_client: TestClient, valid_provider_names: List[str]):
        """
        Test retrieval of provider info for valid provider names.
        """
        for provider_name in valid_provider_names:
            response = test_client.get(f"/aggregation/provider-info/{provider_name}")
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                result = response.json()
                validate_provider_structure(result)
                assert result["provider_name"] == provider_name


class TestEndpointSecurity:

    def test_sql_injection_attempts(self, test_client: TestClient):
        """
        Test common SQL injection patterns are safely handled by endpoints.
        """
        injection_attempts = ["'; DROP TABLE models; --", "1' OR '1'='1", "admin'--"]
        for injection in injection_attempts:
            response = test_client.get(f"/aggregation/provider-info/{injection}")
            assert response.status_code in [400, 404]
            response = test_client.get("/aggregation/compare-models", params={"model_names": [injection]})
            assert response.status_code in [400, 404]


class TestPerformance:

    def test_response_time_reasonable(self, test_client: TestClient):
        """
        Ensure key aggregation endpoints respond within 30 seconds.
        """
        import time
        endpoints = [
            "/aggregation/available-models",
            "/aggregation/provider-comparison"
        ]
        for endpoint in endpoints:
            start_time = time.time()
            response = test_client.get(endpoint)
            end_time = time.time()
            response_time = end_time - start_time
            assert response_time < 30.0, f"Endpoint {endpoint} took too long: {response_time:.2f}s"
            assert response.status_code == 200
