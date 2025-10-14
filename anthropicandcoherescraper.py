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
        "model pricing", "token pricing", "api access", "individual", "team", "enterprise", "api"
    ]

    def __init__(self, urls: List[str], api_token: str = "", output_dir: str = "./output", 
                 delay_between_urls: int = 60, retry_base_delay: int = 15):
        self.urls = urls if isinstance(urls, list) else [urls]
        self.api_token = api_token
        self.output_dir = output_dir
        self.seen_models: Set[str] = set()
        self.delay_between_urls = delay_between_urls
        self.retry_base_delay = retry_base_delay
        os.makedirs(self.output_dir, exist_ok=True)

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
            overlap_rate=0.1,
            apply_chunking=True,
            input_format="markdown",
            extra_args={
                "temperature": 0.05,
                "max_tokens": 5000,
            },
        )

    def _create_deep_crawl_strategy(self):
        """Creates and returns a deep crawling strategy configured to prioritize and filter 
    web pages relevant to AI language models.

    The strategy includes:
    - A `FilterChain` composed of:
        * `URLPatternFilter`: Focuses crawling on URLs containing relevant keywords 
          such as "model", "benchmark", "pricing", "api", "llm", etc.
        * `ContentRelevanceFilter`: Scores content based on its relevance to a 
          predefined query like "AI language model specifications pricing parameters benchmarks"
          with a relevance threshold of 0.3.

    - A `KeywordRelevanceScorer`: Assigns scores to pages based on keyword matches 
      using `self.MODEL_KEYWORDS` with a weight of 0.8.

    The crawling strategy is defined using `BestFirstCrawlingStrategy` with the following parameters:
    - `max_depth=2`: Limits the crawl depth to 2 levels.
    - `include_external=False`: Prevents crawling of external domains.
    - `max_pages=5`: Limits the number of pages to crawl to 5.
    - `filter_chain`: Applies the relevance filters to URLs and content.
    - `url_scorer`: Prioritizes which pages to crawl based on keyword relevance scoring.

    Returns:
        BestFirstCrawlingStrategy: A configured strategy for efficiently exploring 
        web content related to AI models."""
        filter_chain = FilterChain([
            URLPatternFilter(patterns=[
                "*model*", "*benchmark*", "*leaderboard*", "*pricing*", "*api*",
                "*llm*", "*service*", "*product*", "*specification*", "*individual*", 
                "*team*", "*enterprise*"
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
            max_pages=5,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer
        )

    async def _retry_with_timeout(self, func: Callable, retries: int = 5, 
                                 delay: int = None, max_delay: int = 120):
        """ Executes an asynchronous function with retry logic and exponential backoff 
    in case of failure (including timeouts or other exceptions).

    Parameters:
        func (Callable): The asynchronous function to execute.
        retries (int): The maximum number of retry attempts. Default is 5.
        delay (int): The initial delay (in seconds) before retrying. If None, 
                     uses `self.retry_base_delay`.
        max_delay (int): The maximum delay (in seconds) between retries. Default is 120.

    Behavior:
        - On `asyncio.TimeoutError`, it logs the attempt and retries.
        - On any other exception, it logs the error and retries.
        - Uses exponential backoff with jitter: delay doubles on each retry attempt, 
          with added random noise (0–5 seconds), capped by `max_delay`.

    Returns:
        The result of the successful `func()` call, or `None` if all retries fail."""
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
        """saving to json file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_models_extracted_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filepath}")
        return filepath

    def _normalize_model_name(self, name):
        """normalizing names"""
        if not name:
            return ""
        return re.sub(r'[-\s]', '', name.lower())

    def _extract_pricing_info(self, item):
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
        print(f"\n=== Processing URL: {url} ===")
        all_extracted_content = []

        # Step 1: Scrape the initial page with pop-up handling and full scrolling
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
                // Scroll the page with dynamic content detection
                let lastHeight = 0;
                let currentHeight = document.body.scrollHeight;
                let scrollAttempts = 0;
                const maxScrollAttempts = 10;
                while (lastHeight !== currentHeight && scrollAttempts < maxScrollAttempts) {
                    lastHeight = currentHeight;
                    window.scrollTo(0, currentHeight);
                    await new Promise(r => setTimeout(r, 1500));
                    currentHeight = document.body.scrollHeight;
                    scrollAttempts++;
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

        # Step 2: Click on all relevant buttons with dynamic content handling
        try:
            print("Interacting with buttons...")
            button_interaction = """
                // Wait for an element to appear
                async function waitForElement(selector, timeout = 10000) {
                    return new Promise((resolve) => {
                        const start = Date.now();
                        const interval = setInterval(() => {
                            if (document.querySelector(selector)) {
                                clearInterval(interval);
                                resolve(true);
                            } else if (Date.now() - start >= timeout) {
                                clearInterval(interval);
                                resolve(false);
                            }
                        }, 500);
                    });
                }
                // Track clicked elements to prevent duplicates
                const clickedElements = new Set();
                let newWindows = [];
                // Scroll the page with dynamic content detection
                let lastHeight = 0;
                let currentHeight = document.body.scrollHeight;
                let scrollAttempts = 0;
                const maxScrollAttempts = 10;
                while (lastHeight !== currentHeight && scrollAttempts < maxScrollAttempts) {
                    lastHeight = currentHeight;
                    window.scrollTo(0, currentHeight);
                    await new Promise(r => setTimeout(r, 1500));
                    currentHeight = document.body.scrollHeight;
                    scrollAttempts++;
                }
                // Observe DOM for dynamically loaded elements
                const observer = new MutationObserver((mutations) => {
                    mutations.forEach((mutation) => {
                        if (mutation.addedNodes.length) {
                            console.log('New nodes detected, checking for buttons...');
                        }
                    });
                });
                observer.observe(document.body, { childList: true, subtree: true });
                // Wait for potential dynamic buttons
                await new Promise(r => setTimeout(r, 3000));
                // Select a broader range of potential button elements
                const elements = document.querySelectorAll(
                    'button, a, div[role="button"], span[role="button"], ' +
                    '[class*="button"], [class*="btn"], [onclick], [data-action], [data-toggle]'
                );
                const excludeTerms = ['login', 'sign in', 'sign up', 'contact', 'support', 'help'];
                const includeTerms = [
                    'découvrir la tarification', 'explore detailed pricing', 'pricing', 'tarification',
                    'more details', 'individual', 'team', 'enterprise', 'api', 'view plans',
                    'learn more', 'get started', 'details', 'subscriptions', 'buy now', 'purchase'
                ];
                for (const element of elements) {
                    const text = element.textContent.toLowerCase().trim();
                    const elementId = element.id || element.outerHTML.substring(0, 50);
                    if (clickedElements.has(elementId)) {
                        console.log(`Skipped already clicked element: ${element.tagName}, text: "${text}"`);
                        continue;
                    }
                    if (excludeTerms.some(term => text.includes(term))) {
                        console.log(`Skipped irrelevant element: ${element.tagName}, text: "${text}"`);
                        continue;
                    }
                    if (!includeTerms.some(term => text.includes(term))) {
                        console.log(`Skipped non-matching element: ${element.tagName}, text: "${text}"`);
                        continue;
                    }
                    if (!element.disabled && element.getAttribute('aria-disabled') !== 'true') {
                        if (element.tagName.toLowerCase() === 'a' && element.href) {
                            const newWindow = window.open(element.href, '_blank');
                            if (newWindow) {
                                newWindows.push({href: element.href, window: newWindow});
                                clickedElements.add(elementId);
                            } else {
                                console.log(`Failed to open new window for href: ${element.href}`);
                                window.location.href = element.href;
                                await waitForElement('body', 10000);
                                newWindows.push({href: window.location.href, window: window});
                                clickedElements.add(elementId);
                            }
                        } else {
                            element.click();
                            console.log(`Clicked element: ${element.tagName}, text: "${text}"`);
                            clickedElements.add(elementId);
                            await waitForElement('body', 10000);
                            newWindows.push({href: window.location.href, window: window});
                        }
                    } else {
                        console.log(`Skipped disabled element: ${element.tagName}, text: "${text}"`);
                    }
                }
                observer.disconnect();
                window.newWindowsHandles = newWindows;
                window.scrollTo(0, 0);
            """
            pre_crawler_run = await crawler.arun(
                url=url,
                config=pre_crawl_config,
                pre_extraction_js=button_interaction
            )
            async for pre_result in pre_crawler_run:
                pass
            print("Button interactions completed")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Warning: Button interactions encountered an error: {str(e)}")
            await asyncio.sleep(3)

        # Step 3: Extract content from all new pages/windows opened by buttons
        print("Extracting content from new pages...")
        new_window_content = None
        success = False
        error_message = None
        async def extract_new_windows():
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
                new_window_content = []
                js_extract_windows = """
                    if (window.newWindowsHandles && window.newWindowsHandles.length > 0) {
                        for (let i = 0; i < window.newWindowsHandles.length; i++) {
                            window.focus(window.newWindowsHandles[i].window);
                            window.currentWindowUrl = window.newWindowsHandles[i].href;
                        }
                    }
                """
                for window_idx in range(100):
                    crawler_run = await crawler.arun(
                        url=url,
                        config=lxml_config,
                        pre_extraction_js=js_extract_windows + f"if (window.newWindowsHandles && window.newWindowsHandles[{window_idx}]) {{ window.location.href = window.newWindowsHandles[{window_idx}].href; }}"
                    )
                    async for result in crawler_run:
                        if hasattr(result, 'extracted_content'):
                            if isinstance(result.extracted_content, list) and len(result.extracted_content) > 0:
                                new_window_content.extend(result.extracted_content)
                            else:
                                new_window_content.append(result.extracted_content)
                            success = True
                        if hasattr(result, 'error_message'):
                            error_message = result.error_message
                    if not window_idx < len(crawler_run.context.get('newWindowsHandles', [])):
                        break
                return success
            except Exception as e:
                error_message = str(e)
                return False

        await self._retry_with_timeout(extract_new_windows, retries=8, delay=self.retry_base_delay)
        if success and new_window_content:
            all_extracted_content.extend(new_window_content if isinstance(new_window_content, list) else [new_window_content])
            print("New pages extraction successful")
        else:
            print(f"Error extracting new pages: {error_message or 'Failed after retries'}")

        # Process combined data
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
       
        llm_strategy = self._create_llm_strategy()
        deep_crawl_strategy = self._create_deep_crawl_strategy()
        proxy = os.getenv('CRAWLER_PROXY')
        custom_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        browser_cfg = BrowserConfig(
            headless=False,
            verbose=True,
            headers=custom_headers,
            proxy=proxy if proxy else None
        )
        crawl_config = CrawlerRunConfig(
            extraction_strategy=llm_strategy,
            deep_crawl_strategy=deep_crawl_strategy,
            cache_mode=CacheMode.BYPASS,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True,
            verbose=True
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
         "https://www.anthropic.com/pricing",
         "https://cohere.com/pricing"
        
    ]
    crawler = AIModelCrawler(
        urls=URLS_TO_SCRAPE,
        api_token=os.getenv('GROQ_API_TOKEN'),
        output_dir="./output",
        delay_between_urls=12,
        retry_base_delay=20
    )
    result = await crawler.crawl_and_extract()
    return 0 if result else 1

if __name__ == "__main__":
    asyncio.run(main())