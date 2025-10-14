import pytest
import json
import os
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from open_llm_scraper import AIModelCrawler, AIModel, LocalLLMConfig

@pytest.fixture
def crawler():
    """Create a test instance of AIModelCrawler."""
    return AIModelCrawler(
        base_url="https://test.com",
        api_token="test_token",
        output_dir="./test_output",
        max_models=3,
        request_delay=(0.1, 0.2)
    )

@pytest.fixture
def mock_crawler_run():
    """Mock for crawler run results."""
    mock = MagicMock()
    mock.extracted_content = json.dumps({
        "model_name": "TestModel",
        "provider": "TestProvider",
        "parameters": "1B",
        "context_window": "4096",
        "license": "MIT",
        "description": "Test description"
    })
    return mock

class TestAIModelCrawler:
    
    def test_init(self, crawler):
        """Test initialization of AIModelCrawler."""
        assert crawler.base_url == "https://test.com"
        assert crawler.api_token == "test_token"
        assert crawler.output_dir == "./test_output"
        assert crawler.max_models == 3
        assert crawler.min_delay == 0.1
        assert crawler.max_delay == 0.2
        assert crawler.seen_urls == set()
        assert crawler.extracted_models == []
        assert crawler.model_count == 0
    
    @pytest.mark.parametrize("input_data,expected_output", [
        (
            {"model_name": "TestModel", "parameters": "1B"},
            {"model_name": "TestModel", "parameters": 1_000_000_000}
        ),
        (
            {"model_name": "TestModel", "parameters": "1.5B", "context_window": "4096"},
            {"model_name": "TestModel", "parameters": 1_500_000_000, "context_window": 4096}
        ),
        (
            {"model_name": "TestModel", "parameters": 1_000_000_000},
            {"model_name": "TestModel", "parameters": 1_000_000_000}
        ),
        (
            {"model_name": " ModelWithSpace ", "parameters": None},
            {"model_name": "ModelWithSpace", "parameters": None}
        ),
        (
            {"parameters": "1B"}, 
            None
        ),
        (
            {"model_name": "", "parameters": "1B"},  
            None
        ),
    ])
    def test_normalize_model_data(self, crawler, input_data, expected_output):
        """Test model data normalization."""
        result = crawler._normalize_model_data(input_data)
        assert result == expected_output
    
    def test_create_llm_strategy(self, crawler):
        """Test LLM strategy creation."""
        strategy = crawler._create_llm_strategy()
        assert isinstance(strategy, LLMExtractionStrategy)
        assert strategy.llm_config.provider == "groq/meta-llama/llama-4-scout-17b-16e-instruct"
        assert strategy.llm_config.api_token == "test_token"
        assert strategy.input_format == "markdown"
        assert strategy.chunk_token_threshold == 1000
        assert strategy.overlap_rate == 0.2
        assert strategy.apply_chunking is True
        assert "temperature" in strategy.extra_args
        assert "max_tokens" in strategy.extra_args
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("os.path.join", return_value="test_path.json")
    def test_save_results(self, mock_join, mock_json_dump, mock_file, crawler):
        """Test saving results to file."""
        crawler.extracted_models = [{"model_name": "TestModel"}]
        
        result = crawler.save_results(is_final=False)
        assert result == "test_path.json"
        mock_file.assert_called_with("test_path.json", 'w', encoding='utf-8')
        mock_json_dump.assert_called_with([{"model_name": "TestModel"}], mock_file.return_value.__enter__.return_value, indent=2, ensure_ascii=False)
        
        result = crawler.save_results(is_final=True)
        assert result == "test_path.json"
        assert mock_join.call_args[0][1].startswith("ai_models_final_")
    
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_extract_model_urls(self, mock_get, crawler):
        """Test extracting model URLs from API."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [
            {"id": "model1"},
            {"id": "model2"},
            {"id": "model3"}
        ]
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await crawler._extract_model_urls()
        expected_urls = [
            "https://huggingface.co/model1",
            "https://huggingface.co/model2",
            "https://huggingface.co/model3"
        ]
        assert result == expected_urls
        assert crawler.seen_urls == set(expected_urls)
        
        result = await crawler._extract_model_urls()
        assert result == []  
        
        mock_response.status = 500
        result = await crawler._extract_model_urls()
        assert result == []
    
    @pytest.mark.asyncio
    @patch("asyncio.sleep", return_value=None)
    async def test_extract_model_details_success(self, mock_sleep, crawler, mock_crawler_run):
        """Test extracting model details from a URL."""
        mock_crawler = AsyncMock()
        mock_crawler.arun.return_value = mock_crawler_run
        
        result = await crawler._extract_model_details("https://test.com/model", mock_crawler)
        
        assert result is True
        assert len(crawler.extracted_models) == 1
        assert crawler.extracted_models[0]["model_name"] == "TestModel"
        assert crawler.extracted_models[0]["parameters"] == 1_000_000_000
        assert crawler.extracted_models[0]["context_window"] == 4096
    
    @pytest.mark.asyncio
    @patch("asyncio.sleep", return_value=None)
    async def test_extract_model_details_failure(self, mock_sleep, crawler):
        """Test handling extraction failures."""
        mock_crawler = AsyncMock()
        mock_crawler.arun.side_effect = Exception("Test error")
        
        result = await crawler._extract_model_details("https://test.com/model", mock_crawler)
        
        assert result is False
        assert len(crawler.extracted_models) == 0
        assert mock_sleep.call_count == 3  
    
    @pytest.mark.asyncio
    @patch("paste.AIModelCrawler._extract_model_urls")
    @patch("paste.AIModelCrawler._extract_model_details")
    @patch("paste.AIModelCrawler.save_results")
    @patch("paste.save_to_database")
    @patch("asyncio.sleep", return_value=None)
    async def test_crawl_models_success(self, mock_sleep, mock_save_db, mock_save_results,
                                        mock_extract_details, mock_extract_urls, crawler):
        """Test the full crawl process."""
        mock_extract_urls.return_value = ["https://test.com/model1", "https://test.com/model2", "https://test.com/model3"]
        mock_extract_details.side_effect = [True, True, True] 
        mock_save_results.return_value = "test_results.json"
        
        with patch("crawl4ai.AsyncWebCrawler", return_value=AsyncMock()) as mock_web_crawler:
            mock_web_crawler.return_value.__aenter__.return_value = AsyncMock()
            result = await crawler.crawl_models()
        
        assert result == "test_results.json"
        assert mock_extract_details.call_count == 3
        assert mock_save_results.call_count == 2  
        mock_save_db.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("paste.AIModelCrawler._extract_model_urls")
    @patch("asyncio.sleep", return_value=None)
    async def test_crawl_models_no_urls(self, mock_sleep, mock_extract_urls, crawler):
        """Test crawling with no model URLs."""
        mock_extract_urls.return_value = []
        
        with patch("crawl4ai.AsyncWebCrawler", return_value=AsyncMock()) as mock_web_crawler:
            mock_web_crawler.return_value.__aenter__.return_value = AsyncMock()
            result = await crawler.crawl_models()
        
        assert result is None
    
