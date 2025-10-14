
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import math
import numpy as np
from front.helpers import (
    calculate_cost,
    calculate_tokens,
    simulate_model_inference,
    parse_token_unit,
    tokens_for_budget,
)

from app.db.models import Pricing  


@pytest.fixture
def pricing():
    "create a mock pricing object for testing cost calculations"
    return Pricing(
        input_cost=2.0,
        output_cost=4.0,
        cached_input=1.0,
        token_unit="1M",
        currency="USD"
    )


# === TESTS ===

def test_calculate_tokens():
    "test the token calculation logic"
    assert calculate_tokens("abcd") == 1
    assert calculate_tokens("abcdefgh") == 2
    assert calculate_tokens("a" * 20) == 5


def test_calculate_cost_normal(pricing):
    "test the cost calculation with normal input and output tokens"
    cost = calculate_cost(pricing, 500_000, 500_000)
    assert cost == pytest.approx(3.0)


def test_calculate_cost_with_cache(pricing):
    "test the cost calculation with cached input tokens"
    cost = calculate_cost(pricing, 1_000_000, 0, use_cache=True)
    assert cost == 1.0


def test_simulate_model_inference_max_tokens():
    "test the model inference simulation with maximum tokens"
    prompt = "hello world" * 10
    response, in_tokens, out_tokens = simulate_model_inference(prompt, 50)
    assert isinstance(response, str)
    assert in_tokens > 0
    assert out_tokens <= 50


def test_parse_token_unit_million():
    "test the token unit parsing for million tokens"
    assert parse_token_unit("1M") == 1_000_000
    assert parse_token_unit("any_string") == 1_000_000


def test_tokens_for_budget_average_cost():
    "test the token calculation for a given budget with average cost"
    pricings = [
        Pricing(input_cost=1.0, output_cost=3.0, token_unit="1M"),
        Pricing(input_cost=2.0, output_cost=4.0, token_unit="1M")
    ]
    tokens = tokens_for_budget(pricings, 10.0)
    assert tokens > 0
