import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from ai_model_crawler import AIModelCrawler
from unittest.mock import patch
from ai_model_crawler import main
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy

@pytest.fixture
def crawler():
    """
    Fixture to provide an instance of AIModelCrawler with preconfigured parameters.

    Returns:
        AIModelCrawler: An instance of the crawler with predefined URLs, API token, and output directory.
    """
    return AIModelCrawler(
        urls=["https://example.com/test"],
        api_token="test_token",
        output_dir="./test_output"
    )

@pytest.fixture
def test_data():
    """
    Fixture to provide sample test data for extracting models.

    Returns:
        list: A list of dictionaries representing the extracted model data.
    """
    return [
        {
            "model_name": "TestModel-1",
            "provider": "TestOrg",
            "license": "MIT",
            "description": "A test language model",
            "context_window": 4096,
            "max_tokens": 2048,
            "parameters": "7B",
            "arena_score": 95.5,
            "mmlu": 85.2,
            "tasks": "text generation, summarization",
            "input_cost": "$10",
            "output_cost": "$15"
        },
        {
            "model_name": "TestModel-2",
            "organization": "AnotherOrg",  
            "parameters": "70B",
            "ci_95": "±2.3",
            "confidence_interval": "95% CI",
            "context_window": "8192",
            "price_per_token": "$5/1M tokens"
        }
    ]


def test_process_extracted_data(crawler, test_data):
    """
    Test the _process_extracted_data method for correct data processing.

    This test ensures that the extracted data is processed into the correct format
    and that the expected keys and values are present in the processed data.
    """
    processed = crawler._process_extracted_data(test_data)
    
    assert len(processed) == 2
    
    first_model = processed[0]
    assert first_model["model_name"] == "TestModel-1"
    assert first_model.get("benchmarks", {}).get("arena_score") == 95.5
    assert first_model.get("benchmarks", {}).get("mmlu") == 85.2
    assert first_model.get("provider") == "TestOrg"
    assert first_model.get("tasks") == ["text generation", "summarization"]
    assert first_model.get("parameters") == 7 
    assert "pricing" in first_model
    
    second_model = processed[1]
    assert second_model["model_name"] == "TestModel-2"
    assert second_model.get("benchmarks", {}).get("ci_95") == "±2.3"
    assert second_model.get("benchmarks", {}).get("confidence_interval") == "95% CI"
    assert second_model.get("provider") == "AnotherOrg"  
    assert second_model.get("parameters") == 70  
    assert second_model.get("context_window") == 8192 
    assert "organization" not in second_model  

def test_process_extracted_data_empty_input(crawler):
    """
    Test _process_extracted_data with empty or invalid input.
    """
    result = crawler._process_extracted_data([])
    assert result == []
    
    result = crawler._process_extracted_data(None)
    assert result == []
    
    result = crawler._process_extracted_data("invalid")
    assert result == []

def test_process_extracted_data_merge_duplicates(crawler):
    """
    Test that _process_extracted_data correctly merges duplicate models.
    """
    duplicate_data = [
        {
            "model_name": "TestModel",
            "provider": "TestOrg",
            "context_window": 4096,
            "mmlu": 85.0
        },
        {
            "model_name": "TestModel", 
            "provider": "TestOrg",    
            "max_tokens": 2048,       
            "humaneval": 75.0         
        }
    ]
    
    processed = crawler._process_extracted_data(duplicate_data)
    
    assert len(processed) == 1
    model = processed[0]
    assert model["model_name"] == "TestModel"
    assert model["context_window"] == 4096
    assert model["max_tokens"] == 2048
    assert model["benchmarks"]["mmlu"] == 85.0
    assert model["benchmarks"]["humaneval"] == 75.0

def test_normalize_model_name(crawler):
    """
    Test the _normalize_model_name method.
    """
    assert crawler._normalize_model_name("GPT-4") == "gpt4"
    assert crawler._normalize_model_name("Claude 3 Opus") == "claude3opus"
    assert crawler._normalize_model_name("") == ""
    assert crawler._normalize_model_name(None) == ""

def test_extract_pricing_info(crawler):
    """
    Test the _extract_pricing_info method for correct pricing extraction.
    """
    test_item = {
        "model_name": "Test",
        "input_cost": "$10/1M tokens",
        "output_cost": "15.5",
        "price": "$5 per 1K tokens"
    }
    
    result = crawler._extract_pricing_info(test_item)
    
    assert "pricing" in result
    pricing = result["pricing"]
    assert pricing["input_cost"] == 10.0
    assert pricing["output_cost"] == 15.5
    assert pricing["cost"] == 5.0
    assert pricing["currency"] == "$"
    assert "1K tokens" in pricing["unit"]

def test_extract_pricing_info_no_pricing(crawler):
    """
    Test _extract_pricing_info with no pricing information.
    """
    test_item = {"model_name": "Test", "description": "No pricing"}
    result = crawler._extract_pricing_info(test_item)
    assert "pricing" not in result

def test_save_to_json(crawler, test_data, tmp_path):
    """
    Test the save_to_json method for correctly saving extracted data to a JSON file.

    This test ensures that the extracted data is saved to a JSON file in the output directory
    and that the file content matches the expected structure.
    """
    crawler.output_dir = str(tmp_path)
    filename = crawler.save_to_json(test_data, "test_output.json")
    
    file_path = tmp_path / "test_output.json"
    assert file_path.exists()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    assert len(saved_data) == 2
    assert saved_data[0]["model_name"] == "TestModel-1"

def test_save_to_json_default_filename(crawler, test_data, tmp_path):
    """
    Test save_to_json with default filename generation.
    """
    crawler.output_dir = str(tmp_path)
    
    with patch('ai_model_crawler.datetime') as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
        filename = crawler.save_to_json(test_data)
    
    expected_filename = "ai_models_extracted_20240101_120000.json"
    file_path = tmp_path / expected_filename
    assert file_path.exists()

def test_create_llm_strategy(crawler):
    """
    Test the _create_llm_strategy method for generating a valid strategy.

    This test ensures that the LLM strategy is created with a valid schema and
    that the strategy contains the expected properties.
    """
    strategy = crawler._create_llm_strategy()
    assert strategy is not None
    assert isinstance(strategy, LLMExtractionStrategy)
    assert hasattr(strategy, 'schema')
    schema = strategy.schema
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "model_name" in schema["properties"]

def test_create_deep_crawl_strategy(crawler):
    """
    Test the _create_deep_crawl_strategy method.
    """
    strategy = crawler._create_deep_crawl_strategy()
    assert strategy is not None
    
    assert isinstance(strategy, BestFirstCrawlingStrategy)


def test_crawler_initialization():
    """
    Test that AIModelCrawler initializes correctly with various parameters.
    """
    crawler1 = AIModelCrawler("https://test.com", "token")
    assert crawler1.urls == ["https://test.com"]
    
    urls = ["https://test1.com", "https://test2.com"]
    crawler2 = AIModelCrawler(urls, "token")
    assert crawler2.urls == urls
    
    crawler3 = AIModelCrawler(
        urls=["https://test.com"],
        api_token="token",
        output_dir="./custom_output",
        delay_between_urls=30,
        retry_base_delay=10
    )
    assert crawler3.output_dir == "./custom_output"
    assert crawler3.delay_between_urls == 30
    assert crawler3.retry_base_delay == 10
    assert isinstance(crawler3.seen_models, set)

def test_model_keywords():
    """
    Test that MODEL_KEYWORDS constant is properly defined.
    """
    assert hasattr(AIModelCrawler, 'MODEL_KEYWORDS')
    assert isinstance(AIModelCrawler.MODEL_KEYWORDS, list)
    assert len(AIModelCrawler.MODEL_KEYWORDS) > 0
    assert "llm" in AIModelCrawler.MODEL_KEYWORDS

def test_extraction_instruction():
    """
    Test that EXTRACTION_INSTRUCTION constant is properly defined.
    """
    assert hasattr(AIModelCrawler, 'EXTRACTION_INSTRUCTION')
    assert isinstance(AIModelCrawler.EXTRACTION_INSTRUCTION, str)
    assert "model_name" in AIModelCrawler.EXTRACTION_INSTRUCTION
    assert "provider" in AIModelCrawler.EXTRACTION_INSTRUCTION