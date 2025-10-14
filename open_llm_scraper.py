import asyncio
import json
import os
import re
import random
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from pydantic import BaseModel, Field
from app.services.savingdb import save_to_database
from dotenv import load_dotenv
import logging
import aiohttp

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLMConfig:
    def __init__(self, provider: str, api_token: str = "", base_url: str = ""):
        self.provider = provider
        self.api_token = api_token
        self.base_url = base_url

class AIModel(BaseModel):
    """Model schema for AI model data extraction."""
    model_name: str = Field(description="The name of the AI model")
    provider: Optional[str] = Field(description="The name of the model provider", default=None)
    license: Optional[str] = Field(description="License details of the model", default=None)
    description: Optional[str] = Field(description="A brief description of the model", default=None)
    context_window: Optional[int] = Field(description="The context window size for the model", default=None)
    max_tokens: Optional[int] = Field(description="The maximum token limit for the model", default=None)
    benchmarks: Optional[Dict[str, float]] = Field(description="Performance metrics", default=None)
    tasks: Optional[List[str]] = Field(description="Tasks the model is designed for", default=None)
    parameters: Optional[int] = Field(description="Number of parameters in the model", default=None)
    pricing: Optional[Dict[str, Any]] = Field(description="Pricing information for the model", default=None)

class AIModelCrawler:
    EXTRACTION_INSTRUCTION = """
    Extract detailed AI model information from the content. Look for information in model cards, 
    tables, specifications, and technical details.
    
    For each AI model found, extract:
    - model_name: Name of the AI model (e.g., Qwen3-235B-A22B, Dia-1.6B)
    - provider: Organization or author providing the model (e.g., Qwen, deepseek-ai)
    - license: License information if available
    - description: Brief description of the model's capabilities and purpose
    - context_window: Context window size in tokens (convert to integer)
    - parameters: Number of parameters - convert values like "235B" to integer 235
    - benchmarks: Extract all benchmark scores mentioned
    - tasks: List of tasks the model is designed for
    
    Extract ALL relevant information available on the page, even if some fields are missing.
    """

    def __init__(self, base_url: str, api_token: str = "", output_dir: str = "./output", 
                 max_models: int = 50, request_delay: tuple = (3, 8)):
        """Initialize the AI model crawler with necessary configurations."""
        self.base_url = base_url
        self.api_token = api_token
        self.output_dir = output_dir
        self.max_models = max_models
        self.min_delay, self.max_delay = request_delay
        self.seen_urls: Set[str] = set()
        self.extracted_models: List[Dict] = []
        self.model_count = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]

    def _create_llm_strategy(self) -> LLMExtractionStrategy:
        """Creates the LLM extraction strategy for AI model data extraction."""
        return LLMExtractionStrategy(
            llm_config=LocalLLMConfig(
                provider="groq/meta-llama/llama-4-scout-17b-16e-instruct",
                api_token=self.api_token,
                base_url=""
            ),
            schema=AIModel.model_json_schema(),
            extraction_type="schema",
            instruction=self.EXTRACTION_INSTRUCTION,
            chunk_token_threshold=1000,
            overlap_rate=0.2,
            apply_chunking=True,
            input_format="markdown",
            extra_args={"temperature": 0.05, "max_tokens": 4000}
        )

    def save_results(self, is_final=False):
        """Saves the extracted AI model data to a JSON file."""
        prefix = "final" if is_final else "intermediate"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_models_{prefix}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_models, f, indent=2, ensure_ascii=False)
        
        logger.info(f"{prefix.capitalize()} results saved to {filepath}")
        return filepath

    def _normalize_model_data(self, model_data):
        """normalizes the model data by cleaning and converting fields."""
        if not model_data or not model_data.get('model_name'):
            return None
            
        model_data['model_name'] = model_data['model_name'].strip()
        
        if 'parameters' in model_data and model_data['parameters']:
            if isinstance(model_data['parameters'], str):
                param_str = model_data['parameters'].lower()
                if 'b' in param_str:
                    try:
                        value = float(re.sub(r'[^\d.]', '', param_str))
                        model_data['parameters'] = int(value * 1_000_000_000)
                    except ValueError:
                        pass
                        
        if 'context_window' in model_data and model_data['context_window']:
            if isinstance(model_data['context_window'], str):
                try:
                    model_data['context_window'] = int(re.sub(r'[^\d]', '', model_data['context_window']))
                except ValueError:
                    pass
                    
        return model_data

    async def _extract_model_details(self, url, crawler):
        """Extracts detailed AI model information from the given URL using the crawler."""
        logger.info(f"Extracting details from {url}")
        
        llm_strategy = self._create_llm_strategy()
        
        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            cache_mode=CacheMode.BYPASS,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True,
            verbose=True
        )
        
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)
        
        user_agent = random.choice(self.user_agents)
        browser_cfg = BrowserConfig(headless=True, verbose=True, user_agent=user_agent)
        
        detail_page_js = """
        const expandButtons = document.querySelectorAll('button[aria-expanded="false"]');
        for (const button of expandButtons) {
            button.click();
            await new Promise(r => setTimeout(r, 500));
        }
        for (let i = 0; i < 5; i++) {
            window.scrollTo(0, document.body.scrollHeight * (i/5));
            await new Promise(r => setTimeout(r, 300));
        }
        """
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                await crawler.arun(url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS), pre_extraction_js=detail_page_js)
                await asyncio.sleep(2)
                
                crawler_run = await crawler.arun(url=url, config=crawl_config)
                
                extracted_content = getattr(crawler_run, 'extracted_content', None)
                
                if extracted_content:
                    model_data = json.loads(extracted_content) if isinstance(extracted_content, str) else extracted_content
                    
                    if isinstance(model_data, list):
                        for item in model_data:
                            normalized_item = self._normalize_model_data(item)
                            if normalized_item:
                                self.extracted_models.append(normalized_item)
                    else:
                        normalized_item = self._normalize_model_data(model_data)
                        if normalized_item:
                            self.extracted_models.append(normalized_item)
                    
                    if len(self.extracted_models) % 5 == 0:
                        self.save_results(is_final=False)
                    
                    return True
                
                retry_count += 1
                await asyncio.sleep(retry_count * 2)
            
            except Exception as e:
                logger.error(f"Error extracting from {url}: {str(e)}")
                retry_count += 1
                await asyncio.sleep(retry_count * 3)
        
        logger.error(f"Failed to extract from {url} after {max_retries} attempts")
        return False

    async def _extract_model_urls(self):
        """Fetches the list of AI model URLs from the Hugging Face API."""
        api_url = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=20"
        logger.info(f"Fetching model list from API: {api_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    models_data = await response.json()
                    model_urls = [f"https://huggingface.co/{model['id']}" for model in models_data]
                    filtered_urls = [url for url in model_urls if url not in self.seen_urls]
                    self.seen_urls.update(filtered_urls)
                    return filtered_urls
                logger.error(f"API request failed with status {response.status}")
                return []

    async def crawl_models(self):
        """Crawls the Hugging Face models page to extract AI model details."""
        logger.info(f"Starting AI model crawling, max models: {self.max_models}")
        
        browser_cfg = BrowserConfig(headless=True, verbose=True, user_agent=random.choice(self.user_agents))
        
        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                model_urls = await self._extract_model_urls()
                
                if not model_urls:
                    logger.error("No model URLs found")
                    return None
                
                for i, url in enumerate(model_urls):
                    if self.model_count >= self.max_models:
                        break
                    
                    success = await self._extract_model_details(url, crawler)
                    
                    if success:
                        self.model_count += 1
                        if self.model_count % 3 == 0:
                            self.save_results(is_final=False)
                    
                    if i < len(model_urls) - 1:
                        await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
                
                if self.extracted_models:
                    save_to_database(self.extracted_models)
                    return self.save_results(is_final=True)
                return None
                
        except Exception as e:
            logger.error(f"Crawling error: {str(e)}")
            if self.extracted_models:
                return self.save_results(is_final=False)
            return None

async def main():
    """Main function to run the AI model crawler."""
    api_token = os.getenv('GROQ_API_TOKEN')
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    crawler = AIModelCrawler(
        base_url="https://huggingface.co/models",
        api_token=api_token,
        output_dir=output_dir,
        max_models=20,
        request_delay=(3, 8)
    )
    
    result_file = await crawler.crawl_models()
    logger.info(f"Crawling completed: {result_file or 'Failed'}")
    return 0 if result_file else 1

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())