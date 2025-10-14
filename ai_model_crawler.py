import asyncio
import json
import os
from dotenv import load_dotenv
import re
import random
from typing import List, Optional, Dict, Any, Callable, Set
from datetime import datetime
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, ContentRelevanceFilter, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from pydantic import BaseModel, Field
from app.services.savingdb import save_to_database

load_dotenv()

class LocalLLMConfig:
    def __init__(self, provider: str, api_token: str = "", base_url: str = ""):
        self.provider = provider
        self.api_token = api_token
        self.base_url = base_url

class AIModel(BaseModel):
    """
    AI Model schema used for data extraction and storage in line with SQL tables.
    """
    model_name: str = Field(description="The name of the AI model")
    provider: Optional[str] = Field(description="The name of the model provider", default=None)
    provider_id: Optional[int] = Field(description="The ID of the provider (foreign key)", default=None)
    license: Optional[str] = Field(description="License details of the model", default=None)
    description: Optional[str] = Field(description="A brief description of the model", default=None)
    context_window: Optional[int] = Field(description="The context window size for the model", default=None)
    max_tokens: Optional[int] = Field(description="The maximum token limit for the model", default=None)
    benchmarks: Optional[Dict[str, float]] = Field(description="Performance metrics (e.g., Average, MMLU, etc.)", default=None)
    tasks: Optional[List[str]] = Field(description="Tasks the model is designed for", default=None)
    parameters: Optional[int] = Field(description="Number of parameters in the model", default=None)
    pricing: Optional[Dict[str, Any]] = Field(description="Pricing information for the model", default=None)

class AIModelCrawler:
    """
    A crawler to scrape AI model data from given URLs and extract relevant details.
    """
    EXTRACTION_INSTRUCTION = """
    Extract AI model information from the content, focusing on models mentioned on the page. 
    Look for information in tables, text descriptions, and pricing sections.
    For each AI model found, extract:
    - model_name: Name of the AI model (e.g., GPT-4, Claude 3 Opus)
    - provider: Organization providing the model (e.g., OpenAI, Anthropic)
    - license: License details if available
    - description: Brief description of the model's capabilities
    - context_window: Context window size in tokens (integer)
    - max_tokens: Maximum tokens the model can process (integer)
    - parameters: Number of parameters - convert values like "70B" to integer 70
    - benchmarks: Performance metrics like MMLU, GPQA, HumanEval, etc.
      - name: Name of the benchmark (e.g., humanEval)
      - score_benchmark: Numerical score achieved for the specific benchmark
    - confidence_interval: Average 95% confidence interval (if available) as a string
    - votes: Average number of votes (if available) in the benchmark
    - tasks: List of tasks the model is designed for
    - pricing: Extract pricing information including input_cost, output_cost, cached_input, training_cost, cost, unit, and currency
    Extract ALL models mentioned on the page, even if information is incomplete.
    If no specific models are mentioned, extract general pricing information and assign it to a placeholder model name like "OpenAI API General".
    """

    MODEL_KEYWORDS = [
        "llm", "language model", "ai model", "gpt", "claude", "llama", "mistral",
        "gemini", "leaderboard", "benchmark", "context window", "parameters", 
        "model pricing", "token pricing", "api access"
    ]

    def __init__(self, urls: List[str], api_token: str = "", output_dir: str = "./output", 
                 delay_between_urls: int = 60, retry_base_delay: int = 15):
        """Initialize the AIModelCrawler with URLs and configuration parameters."""
        self.urls = urls if isinstance(urls, list) else [urls]
        self.api_token = api_token
        self.output_dir = output_dir
        self.seen_models: Set[str] = set()
        self.delay_between_urls = delay_between_urls
        self.retry_base_delay = retry_base_delay
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_llm_strategy(self) -> LLMExtractionStrategy:
        """Create the LLM extraction strategy for extracting AI model data."""
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
            overlap_rate=0.1,
            apply_chunking=True,
            input_format="markdown",
            extra_args={
                "temperature": 0.05,
                "max_tokens": 5000,
            },
        )

    def _create_deep_crawl_strategy(self):
        """Create the deep crawling strategy for extracting AI model data."""
        filter_chain = FilterChain([
            URLPatternFilter(patterns=[
                "*model*", "*benchmark*", "*leaderboard*", "*pricing*", "*api*",
                "*llm*", "*service*", "*product*", "*specification*"
            ]),
            ContentRelevanceFilter(
                query="AI language model specifications pricing parameters benchmarks",
                threshold=0.3
            )
        ])
        keyword_scorer = KeywordRelevanceScorer(
            keywords=self.MODEL_KEYWORDS,
            weight=0.8
        )
        return BestFirstCrawlingStrategy(
            max_depth=2,
            include_external=False,
            max_pages=2,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        )

    async def _retry_with_timeout(self, func: Callable, retries: int = 5, 
                                 delay: int = None, max_delay: int = 120):
        """Retry a function with exponential backoff and timeout handling."""
        
        delay = delay or self.retry_base_delay
        for attempt in range(retries):
            try:
                return await func()
            except asyncio.TimeoutError:
                print(f"Attempt {attempt + 1} failed with timeout.")
            except Exception as e:
                print(f"Error in attempt {attempt + 1}: {str(e)}")
            backoff_time = min(delay * (2 ** attempt) + random.uniform(0, 5), max_delay)
            print(f"Retrying in {backoff_time:.1f} seconds (attempt {attempt+1}/{retries})...")
            await asyncio.sleep(backoff_time)
        print("All retries failed.")
        return None

    def save_to_json(self, data, filename=None):
        """ Save data to a JSON file in the specified output directory.

    If no filename is provided, a default filename will be generated
    using the current timestamp in the format `ai_models_extracted_YYYYMMDD_HHMMSS.json`.
"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_models_extracted_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filepath}")
        return filepath

    def _normalize_model_name(self, name):
        """normalize the names of llms"""
        if not name:
            return ""
        return re.sub(r'[-\s]', '', name.lower())

    def _extract_pricing_info(self, item):
        """Extract pricing-related information from a given item and standardize it into a structured format.

    This method looks for common pricing-related keys (e.g., 'input_cost', 'output_cost', 'price', etc.)
    in the `item` dictionary. If it finds any, it attempts to:
    - Parse the value (as float, if string with currency symbols or units),
    - Detect the currency symbol (e.g., $, €, £, ¥),
    - Extract the unit (e.g., "per 1K tokens").

    The result is stored in a new nested dictionary under `item['pricing']`, with keys:
    - 'input_cost', 'output_cost', 'training_cost', 'cached_input', 'cost'
    - 'currency': currency symbol (default "USD" if not found),
    - 'unit': pricing unit (default "token" if not found).

    Args:
        item (dict): A dictionary potentially containing raw pricing fields.

    Returns:
        dict: The original `item` dictionary with an added 'pricing' field if pricing info was found."""
        pricing = {}
        price_keys = ['input_cost', 'output_cost', 'price_per_token', 'price', 'cached_input', 'training_cost', 'cost']
        for key in price_keys:
            if key in item and item[key]:
                if isinstance(item[key], (int, float)):
                    value = float(item[key])
                elif isinstance(item[key], str):
                    price_str = item[key].strip()
                    currency_match = re.search(r'[$€£¥]', price_str)
                    currency = currency_match.group(0) if currency_match else "USD"
                    num_match = re.search(r'[\d.,]+', price_str)
                    value = float(num_match.group(0).replace(',', '')) if num_match else None
                    unit_match = re.search(r'per\s+([\d\s\w]+(?:\s+[^\s]+)?)', price_str, re.IGNORECASE)
                    unit = unit_match.group(1).strip() if unit_match else "token"
                    if value is not None:
                        if 'input_cost' in key:
                            pricing['input_cost'] = value
                        elif 'output_cost' in key:
                            pricing['output_cost'] = value
                        elif 'cached_input' in key:
                            pricing['cached_input'] = value
                        elif 'training_cost' in key:
                            pricing['training_cost'] = value
                        else:
                            pricing['cost'] = value
                        pricing['currency'] = currency
                        pricing['unit'] = unit
        if pricing:
            item['pricing'] = pricing
        return item

    def _process_extracted_data(self, raw_data):
        """ Process and normalize raw model data into a structured, deduplicated format.

    This function performs the following steps:
    1. Converts input into a list of dictionaries (if needed).
    2. Normalizes each model entry (e.g., trims names, restructures benchmark fields).
    3. Extracts pricing info using `_extract_pricing_info`.
    4. Converts parameter counts (e.g., "7B", "800M") into integer values in billions.
    5. Cleans and converts numeric fields like context_window or max_tokens.
    6. Splits `tasks` strings into lists.
    7. Merges entries that refer to the same model-provider pair.

    Args:
        raw_data (Union[dict, list]): Raw model data extracted from a source.

    Returns:
        List[dict]: A list of cleaned and merged model entries."""
        processed_data = []
        if not isinstance(raw_data, list):
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            else:
                print("Warning: Extracted content is not a list or dict")
                return []
        for item in raw_data:
            if not item:
                continue
            model_data = dict(item)
            if not model_data.get('model_name'):
                continue
            model_data['model_name'] = model_data['model_name'].strip()
            if not model_data.get('benchmarks'):
                model_data['benchmarks'] = {}
            benchmark_fields = ['arena_score', 'mmlu', 'gsm8k', 'math', 'truthfulqa', 'humaneval']
            for field in benchmark_fields:
                if field in model_data and model_data[field]:
                    model_data['benchmarks'][field] = model_data[field]
                    del model_data[field]
            if 'ci_95' in model_data and model_data['ci_95']:
                model_data['benchmarks']['ci_95'] = model_data['ci_95']
                del model_data['ci_95']
            if 'confidence_interval' in model_data and model_data['confidence_interval']:
                model_data['benchmarks']['confidence_interval'] = model_data['confidence_interval']
                del model_data['confidence_interval']
            if 'organization' in model_data and not model_data.get('provider'):
                model_data['provider'] = model_data['organization']
                if 'organization' != 'provider':
                    del model_data['organization']
            if 'parameters' in model_data and model_data['parameters']:
                if isinstance(model_data['parameters'], str):
                    param_str = model_data['parameters'].lower()
                    if 'b' in param_str:
                        try:
                            model_data['parameters'] = int(float(param_str.replace('b', '').strip()))
                        except ValueError:
                            pass
                    elif 'm' in param_str:
                        try:
                            model_data['parameters'] = int(float(param_str.replace('m', '').strip()) / 1000)
                        except ValueError:
                            pass
            for field in ['context_window', 'max_tokens']:
                if field in model_data and model_data[field] and isinstance(model_data[field], str):
                    num_match = re.search(r'[\d,]+', model_data[field])
                    if num_match:
                        try:
                            model_data[field] = int(num_match.group(0).replace(',', ''))
                        except ValueError:
                            pass
            if 'tasks' in model_data and model_data['tasks']:
                if isinstance(model_data['tasks'], str):
                    model_data['tasks'] = [t.strip() for t in re.split(r'[,;|]', model_data['tasks'])]
            model_data = self._extract_pricing_info(model_data)
            processed_data.append(model_data)
        merged_data = []
        seen_models = {}
        for item in processed_data:
            norm_name = self._normalize_model_name(item['model_name'])
            provider = item.get('provider', '')
            norm_provider = provider.lower().strip() if provider else ""
            key = (norm_name, norm_provider)
            if key in seen_models:
                existing = seen_models[key]
                for k, v in item.items():
                    if v is not None and (k not in existing or existing[k] is None):
                        existing[k] = v
                    elif isinstance(v, dict) and isinstance(existing.get(k), dict):
                        for sub_k, sub_v in v.items():
                            if sub_v is not None and (sub_k not in existing[k] or existing[k][sub_k] is None):
                                existing[k][sub_k] = sub_v
                    elif isinstance(v, list) and isinstance(existing.get(k), list):
                        try:
                            existing[k] = list(set(existing[k] + v))
                        except TypeError:
                            combined = existing[k].copy()
                            for val in v:
                                if val not in combined:
                                    combined.append(val)
                            existing[k] = combined
            else:
                seen_models[key] = item
                merged_data.append(item)
        return merged_data

    async def _process_url(self, url, crawler, crawl_config):
        """Process a single URL by loading the page, extracting content, and handling pop-ups."""
        print(f"\n=== Processing URL: {url} ===")
        all_extracted_content = []

        try:
            print("Loading and interacting with the initial page...")
            pre_crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            initial_interaction = """
                // Handle pop-up
                const buttons = document.querySelectorAll('button, [role="button"]');
                for (const button of buttons) {
                    if (button.textContent.toLowerCase().includes('ok')) {
                        button.click();
                        await new Promise(r => setTimeout(r, 1000));
                        break;
                    }
                }
                // Scroll the page
                let lastHeight = 0;
                let currentHeight = document.body.scrollHeight;
                while (lastHeight !== currentHeight) {
                    lastHeight = currentHeight;
                    window.scrollTo(0, currentHeight);
                    await new Promise(r => setTimeout(r, 1000));
                    currentHeight = document.body.scrollHeight;
                }
                window.scrollTo(0, 0);
            """
            pre_crawler_run = await crawler.arun(
                url=url,
                config=pre_crawl_config,
                pre_extraction_js=initial_interaction
            )
            async for pre_result in pre_crawler_run:
                pass
            print("Initial page pre-processing completed")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Warning: Initial pre-processing encountered an error: {str(e)}")
            await asyncio.sleep(3)

        print("Extracting content from initial page...")
        extracted_content = None
        success = False
        error_message = None
        async def extract_initial():
            """Extract content from the initial page."""
            nonlocal extracted_content, success, error_message
            try:
                lxml_config = CrawlerRunConfig(
                    extraction_strategy=crawl_config.extraction_strategy,
                    deep_crawl_strategy=crawl_config.deep_crawl_strategy,
                    cache_mode=CacheMode.BYPASS,
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    stream=True,
                    verbose=True
                )
                crawler_run = await crawler.arun(url=url, config=lxml_config)
                async for result in crawler_run:
                    if hasattr(result, 'extracted_content'):
                        if isinstance(result.extracted_content, list) and len(result.extracted_content) > 0:
                            if extracted_content is None:
                                extracted_content = result.extracted_content
                            else:
                                extracted_content.extend(result.extracted_content)
                        else:
                            extracted_content = result.extracted_content
                        success = True
                    if hasattr(result, 'error_message'):
                        error_message = result.error_message
                return success
            except Exception as e:
                error_message = str(e)
                return False

        await self._retry_with_timeout(extract_initial, retries=8, delay=self.retry_base_delay)
        if success and extracted_content:
            all_extracted_content.extend(extracted_content if isinstance(extracted_content, list) else [extracted_content])
            print("Initial page extraction successful")
        else:
            print(f"Error extracting initial page: {error_message or 'Failed after retries'}")

        try:
            print("Interacting with button...")
            button_interaction = """
                let newWindow = null;
                let lastHeight = 0;
                let currentHeight = document.body.scrollHeight;
                while (lastHeight !== currentHeight) {
                    lastHeight = currentHeight;
                    window.scrollTo(0, currentHeight);
                    await new Promise(r => setTimeout(r, 1000));
                    currentHeight = document.body.scrollHeight;
                }
                const elements = document.querySelectorAll('button, a, div[role="button"]');
                for (const element of elements) {
                    const text = element.textContent.toLowerCase();
                    if (text.includes('découvrir la tarification') || 
                        text.includes('explore detailed pricing') || 
                        text.includes('pricing') || 
                        text.includes('tarification') || 
                        text.includes('more details')) {
                        if (element.tagName.toLowerCase() === 'a' && element.href) {
                            newWindow = window.open(element.href, '_blank');
                        } else {
                            newWindow = window.open('', '_blank');
                            element.click();
                        }
                        await new Promise(r => setTimeout(r, 2000));
                        break;
                    }
                }
                if (newWindow) {
                    newWindow.document.write(document.documentElement.outerHTML);
                    window.newWindowHandle = newWindow;
                }
                window.scrollTo(0, 0);
            """
            pre_crawler_run = await crawler.arun(
                url=url,
                config=pre_crawl_config,
                pre_extraction_js=button_interaction
            )
            async for pre_result in pre_crawler_run:
                pass
            print("Button interaction completed")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Warning: Button interaction encountered an error: {str(e)}")
            await asyncio.sleep(3)

        print("Extracting content from new page...")
        new_window_content = None
        success = False
        error_message = None
        async def extract_new_window():
            """ Asynchronously extracts content from a newly opened browser window or tab using an LXML-based crawler.

    This function uses a `CrawlerRunConfig` with:
    - LXML scraping (faster, static HTML-based)
    - Cache bypassing (force fresh content)
    - Streamed crawling (asynchronous result yielding)

    A custom JavaScript is injected to focus the new window if its handle is available in the JS context.
    Extracted content is collected into the `new_window_content` list, and status is tracked with `success` and `error_message`.

    This function is meant to be passed to a retry wrapper: `self._retry_with_timeout(...)`
"""
            nonlocal new_window_content, success, error_message
            try:
                lxml_config = CrawlerRunConfig(
                    extraction_strategy=crawl_config.extraction_strategy,
                    deep_crawl_strategy=crawl_config.deep_crawl_strategy,
                    cache_mode=CacheMode.BYPASS,
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    stream=True,
                    verbose=True
                )
                crawler_run = await crawler.arun(
                    url=url,
                    config=lxml_config,
                    pre_extraction_js="if (window.newWindowHandle) { window.focus(window.newWindowHandle); }"
                )
                async for result in crawler_run:
                    if hasattr(result, 'extracted_content'):
                        if isinstance(result.extracted_content, list) and len(result.extracted_content) > 0:
                            if new_window_content is None:
                                new_window_content = result.extracted_content
                            else:
                                new_window_content.extend(result.extracted_content)
                        else:
                            new_window_content = result.extracted_content
                        success = True
                    if hasattr(result, 'error_message'):
                        error_message = result.error_message
                return success
            except Exception as e:
                error_message = str(e)
                return False

        await self._retry_with_timeout(extract_new_window, retries=8, delay=self.retry_base_delay)
        if success and new_window_content:
            all_extracted_content.extend(new_window_content if isinstance(new_window_content, list) else [new_window_content])
            print("New page extraction successful")
        else:
            print(f"Error extracting new page: {error_message or 'Failed after retries'}")

        if all_extracted_content:
            try:
                raw_data = []
                for content in all_extracted_content:
                    if isinstance(content, str):
                        try:
                            raw_data.extend(json.loads(content))
                        except json.JSONDecodeError:
                            print("Warning: Content is text, reinterpreting...")
                            wrapped_content = f'{{"model_name": "Unknown", "description": {json.dumps(content)}}}'
                            raw_data.append(json.loads(wrapped_content))
                    else:
                        raw_data.append(content)
                processed_data = self._process_extracted_data(raw_data)
                print(f"Successfully extracted {len(processed_data)} models from {url}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                domain = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1) or "unknown"
                intermediate_filename = f"intermediate_{domain}_{timestamp}.json"
                intermediate_path = os.path.join(self.output_dir, intermediate_filename)
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                print(f"Saved intermediate results to {intermediate_path}")
                return processed_data
            except Exception as e:
                print(f"Error processing extracted content: {str(e)}")
                return []
        else:
            print("No content extracted from either page.")
            return []

    async def crawl_and_extract(self):
        """  Asynchronously crawls and extracts AI model information from a list of URLs.

    This method initializes a web crawler with LLM-based extraction and deep crawling strategies,
    iterates over the URLs provided in `self.urls`, and performs scraping and structured extraction.
    It respects a configurable delay between requests to prevent rate limiting.

    For each URL, it extracts models, aggregates them into a master list, and logs the number
    of models found per URL. If models are successfully extracted:
      - A diagnostic JSON file summarizing the crawl is written to `self.output_dir`.
      - The extracted models are saved to the database.
      - The models are also saved to a JSON file via `self.save_to_json`.
"""
        
        llm_strategy = self._create_llm_strategy()
        deep_crawl_strategy = self._create_deep_crawl_strategy()
        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            deep_crawl_strategy=deep_crawl_strategy,
            cache_mode=CacheMode.BYPASS,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True,
            verbose=True
        )
        browser_cfg = BrowserConfig(
            headless=True,
            verbose=True,
        )
        all_models = []
        url_results = {}
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            for i, url in enumerate(self.urls):
                if i > 0:
                    delay = self.delay_between_urls + random.uniform(0, 10)
                    print(f"\nWaiting {delay:.1f} seconds before processing next URL to avoid rate limits...")
                    await asyncio.sleep(delay)
                models = await self._process_url(url, crawler, crawl_config)
                url_results[url] = len(models) if models else 0
                if models:
                    all_models.extend(models)
                    print(f"Added {len(models)} models from {url} to the collection")
                else:
                    print(f"No models were extracted from {url}")
        if all_models:
            diag_filename = f"crawl_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            diag_path = os.path.join(self.output_dir, diag_filename)
            with open(diag_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "crawl_time": datetime.now().isoformat(),
                    "total_models_extracted": len(all_models),
                    "urls_processed": len(self.urls),
                    "results_by_url": url_results
                }, f, indent=2)
            save_to_database(all_models)
            output_file_json = self.save_to_json(all_models)
            print(f"\nCrawling completed! Successfully extracted {len(all_models)} AI models from {len(self.urls)} URLs.")
            print(f"Extraction summary by URL: {url_results}")
            return output_file_json
        else:
            print("Error: No AI models were extracted.")
            return None

async def main():
    URLS_TO_SCRAPE = [
        #"https://openai.com/fr-FR/api/pricing/",
         "https://www.vellum.ai/llm-leaderboard",
        #"https://groq.com/pricing/"

    ]
    crawler = AIModelCrawler(
        urls=URLS_TO_SCRAPE,
        api_token=os.getenv('GROQ_API_TOKENSSS'),
        output_dir="./output",
        delay_between_urls=12,
        retry_base_delay=20
    )
    result = await crawler.crawl_and_extract()
    return 0 if result else 1

if __name__ == "__main__":
    asyncio.run(main())