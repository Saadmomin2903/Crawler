import asyncio
import json
import os
import random
import time
import urllib
from collections import defaultdict
from typing import Set
from urllib.parse import urlparse
from urllib.parse import urlparse

import aioredis
import pandas as pd
import redis.asyncio as redis
import robotexclusionrulesparser
import simhash
import spacy
import structlog
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError
from pydantic_settings import BaseSettings
from pygments.lexers import TextLexer
from starlette.responses import JSONResponse
from structlog.stdlib import BoundLogger

# Load environment variables
load_dotenv()

# Logging configuration
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=BoundLogger,  # Changed from structlog.BoundLogger to BoundLogger
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


class Settings(BaseSettings):
    MAX_DEPTH: int = 5
    START_URL: str = ""
    CONCURRENT_REQUESTS: int = 500
    RETRY_LIMIT: int = 3
    DELAY_RANGE: tuple = (1, 3)
    MAX_CRAWL_TIME: int = 7200  # 2 hours
    UPSTASH_REDIS_URL: str = os.getenv('UPSTASH_REDIS_URL')
    UPSTASH_REDIS_TOKEN: str = os.getenv('UPSTASH_REDIS_TOKEN')
    MIN_INSTANCES: int = 5
    MAX_INSTANCES: int = 15
    SCALE_UP_THRESHOLD: int = 1000
    SCALE_DOWN_THRESHOLD: int = 400
    BATCH_SIZE: int = 200


settings = Settings()

redis_client = redis.from_url(
    settings.UPSTASH_REDIS_URL,
    password=settings.UPSTASH_REDIS_TOKEN,
    decode_responses=True
)
from pydantic import BaseModel
from fastapi import HTTPException


class CrawlSettings(BaseModel):
    start_url: str
    max_depth: int = 5
    concurrent_requests: int = 500
    retry_limit: int = 3
    delay_range: tuple = (1, 3)
    max_crawl_time: int = 7200


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:63342"],  # Add your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_base_domain(url: str) -> str:
    return urlparse(url).netloc


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([a-zA-Z])-([a-zA-Z])', r'\1 - \2', text)
    return text


from typing import Dict, List, Union
from bs4 import BeautifulSoup


def extract_content_from_soup(soup: BeautifulSoup, base_url: str, strategy: str = 'advanced') -> Dict[
    str, Union[str, List[str]]]:
    if strategy == 'basic':
        return extract_basic_content(soup)
    elif strategy == 'custom':
        return extract_custom_content(soup)
    else:
        return extract_advanced_content(soup, base_url)


def extract_basic_content(soup: BeautifulSoup) -> Dict[str, str]:
    """Basic content extraction strategy."""
    title = clean_text(soup.title.string) if soup.title else "No Title"
    return {
        "title": title
    }


def extract_custom_content(soup: BeautifulSoup) -> Dict[str, str]:
    """Custom content extraction strategy."""
    custom_content = []

    # Extract paragraphs
    paragraphs = soup.find_all('p', class_='article-text')
    custom_content.extend([clean_text(p.get_text()) for p in paragraphs])

    # Extract headers
    headers = soup.find_all(['h1', 'h2', 'h3'], class_='header-title')
    custom_content.extend([clean_text(h.get_text()) for h in headers])

    # Extract main content divs
    divs = soup.find_all('div', class_='main-content')
    custom_content.extend([clean_text(div.get_text()) for div in divs])

    return {
        "title": clean_text(soup.title.string) if soup.title else "No Title",
        "content": custom_content
    }


import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

import spacy
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin
from typing import Dict, Union, List

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_advanced_content(soup: BeautifulSoup, base_url: str) -> Dict[str, Union[str, List[str]]]:
    """
    Extract content from BeautifulSoup object with advanced handling for different HTML elements.
    Excludes non-content elements and handles code blocks and embedded content.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML.
        base_url (str): The base URL for resolving relative links.

    Returns:
        Dict[str, Union[str, List[str]]]: Extracted content including title and main content blocks.
    """

    exclude_tags = {'script', 'style', 'noscript', 'object', 'embed', 'nav', 'aside', 'footer', 'button',
                    'svg', 'form', 'textarea', 'select', 'a'}
    exclude_classes = {'nav', 'navbar', 'header', 'footer', 'sidebar', 'menu'}
    exclude_ids = {'footer', 'sidebar', 'menu', 'button', 'navbar', 'nav'}

    def clean_url(url: str) -> str:
        """Resolve relative URLs to absolute URLs."""
        if not url.startswith(('http://', 'https://', '#', 'javascript:')):
            url = urljoin(base_url, url)
        return url

    def extract_title() -> str:
        """Extract the title from the soup."""
        return clean_text(soup.title.string) if soup.title else "No Title"

    def extract_main_content() -> List[str]:
        """Extract meaningful content from the main section of the soup."""
        content_blocks = []
        seen_content = set()
        main_content = soup.find('main') or soup.find('article') or soup.body

        if not main_content:
            return content_blocks

        for element in main_content.descendants:
            if isinstance(element, NavigableString):
                continue

            if element.name in exclude_tags:
                continue
            if getattr(element, 'class', []) and any(cls in exclude_classes for cls in element.get('class')):
                continue
            if getattr(element, 'id', '') in exclude_ids:
                continue

            extracted_content = None

            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                header_level = int(element.name[1])
                extracted_content = f"{'#' * header_level} {clean_text(element.get_text())}"


            elif element.name == 'p':
                p_text = clean_text(element.get_text())
                if p_text:
                    extracted_content = p_text

            elif element.name in ['ul', 'ol']:
                list_items = [f"- {clean_text(li.get_text())}" for li in element.find_all('li', recursive=False)]
                if list_items:
                    extracted_content = "".join(list_items)

            elif element.name == 'blockquote':
                quote_text = clean_text(element.get_text())
                if quote_text:
                    extracted_content = f"> {quote_text}"

            elif element.name in ['pre', 'code']:
                code_text = element.get_text(strip=True)
                language = element.get('class', [''])[0].split('-')[-1]
                extracted_content = f"```{language}\n{code_text}\n```"

            elif element.name == 'a':
                link_url = clean_url(element.get('href', ''))
                link_text = clean_text(element.get_text())
                if link_url and link_text and get_base_domain(link_url) == get_base_domain(base_url):
                    extracted_content = f"[{link_text}]({link_url})"

            elif element.name == 'img':
                img_src = clean_url(element.get('src', ''))
                img_alt = clean_text(element.get('alt', ''))
                img_title = clean_text(element.get('title', ''))
                if img_src:
                    img_text = f"![{img_alt}]({img_src})"
                    if img_title:
                        img_text += f' "{img_title}"'
                    extracted_content = img_text

            elif element.name == 'video':
                video_src = clean_url(
                    element.get('src') or (element.find('source') and element.find('source').get('src')))
                video_caption = clean_text(element.get('alt', ''))
                if video_src:
                    video_text = f"[Video]({video_src})"
                    if video_caption:
                        video_text += f' "{video_caption}"'
                    extracted_content = video_text

            elif element.name == 'audio':
                audio_src = clean_url(
                    element.get('src') or (element.find('source') and element.find('source').get('src')))
                audio_caption = clean_text(element.get('alt', ''))
                if audio_src:
                    audio_text = f"[Audio]({audio_src})"
                    if audio_caption:
                        audio_text += f' "{audio_caption}"'
                    extracted_content = audio_text

            elif element.name == 'iframe':
                iframe_src = clean_url(element.get('src', ''))
                if iframe_src:
                    extracted_content = f"[Embedded Content]({iframe_src})"

            elif element.name == 'table':
                table_rows = []
                headers = [clean_text(th.get_text()) for th in element.find_all('th')]
                if headers:
                    table_rows.append(" | ".join(headers))
                    table_rows.append("--- | " * len(headers))

                for row in element.find_all('tr'):
                    cells = [clean_text(cell.get_text()) for cell in row.find_all(['th', 'td'])]
                    table_rows.append(' | '.join(cells))
                if table_rows:
                    table_text = " ".join(table_rows)
                    extracted_content = f"{table_text}"

            elif element.name == 'dl':
                terms_definitions = []
                for dt in element.find_all('dt'):
                    term = clean_text(dt.get_text())
                    definition = clean_text(dt.find_next_sibling('dd').get_text())
                    terms_definitions.append(f"**{term}**: {definition}")
                if terms_definitions:
                    extracted_content = " ".join(terms_definitions)

            if extracted_content and extracted_content not in seen_content:
                seen_content.add(extracted_content)
                content_blocks.append(extracted_content)

        return content_blocks

    title = extract_title()
    main_content = extract_main_content()
    return {
        "title": title,
        "content": main_content,
        "processedContent": main_content
    }


def convert_processed_content(processed_content: List[str]) -> str:
    """
    Convert processed content into a clean and readable string with advanced processing,
    removing duplicates and normalizing whitespace. Formats code blocks and converts tables to DataFrames.

    Args:
        processed_content (List[str]): List of processed content strings.

    Returns:
        str: A single string containing the cleaned and formatted content.
    """
    seen = set()
    unique_content = []

    for item in processed_content:
        stripped_item = item.strip()

        # Check for and handle code blocks
        if stripped_item.startswith('```') and stripped_item.endswith('```'):
            code_content = stripped_item[3:-3].strip()
            language = code_content.split(' ')[0].strip()
            code = ' '.join(code_content.split(' ')[1:])
            processed_text = format_code_block(code, language)
        # Check for and handle tables
        elif stripped_item.startswith('|'):
            try:
                df = extract_table_to_dataframe(stripped_item)
                processed_text = f"<table>{df.to_html(index=False)}</table>"
            except Exception as e:
                processed_text = f"<table_error>{str(e)}</table_error>"
        # Check for and handle images, videos, audio, and links
        elif any(marker in stripped_item for marker in ['![Image]', '[Video]', '[Audio]', '](http']):
            processed_text = stripped_item
        else:
            # Process text with spaCy
            doc = nlp(stripped_item)
            processed_text = ' '.join([
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct
            ])
            # Normalize whitespace
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        if processed_text and processed_text not in seen:
            seen.add(processed_text)
            unique_content.append(processed_text)

    return ' '.join(unique_content)


def extract_images(soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
    """
    Extract all image URLs from the BeautifulSoup object and return them as a list of dictionaries.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML.
        base_url (str): The base URL for resolving relative links.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the 'src' and 'alt' of an image.
    """
    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        alt = img.get("alt", "")

        # Resolve relative URLs
        if src and not src.startswith(("http://", "https://")):
            src = urljoin(base_url, src)

        if src:
            images.append({"src": src, "alt": alt})

    return images


async def extract_links_from_soup(soup: BeautifulSoup, base_url: str) -> Set[str]:
    """Extract and normalize links from BeautifulSoup object."""
    base_domain = get_base_domain(base_url)
    links = set()
    for link in soup.find_all("a", href=True):
        href = link['href']
        if not href.startswith(('http://', 'https://', '#', 'javascript:')):
            href = urljoin(base_url, href)
        if get_base_domain(href) == base_domain:
            links.add(href)
    return links


import spacy
import re
from typing import List

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter


def get_lexer(language):
    language_map = {
        'npm': 'shell',
        'ChatBox': 'python',  # Assuming ChatBox is Python-related
        # Add more mappings as needed
    }

    mapped_language = language_map.get(language, language)

    try:
        return get_lexer_by_name(mapped_language, stripall=True)
    except ValueError:
        return TextLexer()


def format_code_block(code, language):
    lexer = get_lexer(language)
    formatter = HtmlFormatter()
    return highlight(code, lexer, formatter)


def extract_table_to_dataframe(table_html: str) -> pd.DataFrame:
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    return pd.read_html(str(table))[0]


class Crawler:
    def __init__(self):
        self.settings = Settings()
        self.visited = set()
        self.uncrawled_links = set()
        self.failed_links = {}
        self.queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(settings.CONCURRENT_REQUESTS)
        self.start_time = time.time()
        self.error_counts = {}
        self.content_fingerprints = set()
        self.current_instances = settings.MIN_INSTANCES
        self.batch_queue = asyncio.Queue()
        self.processing_queue = asyncio.Queue()
        self.concurrent_requests = settings.CONCURRENT_REQUESTS
        self.rate_limit_delay = 1  # Initial delay of 1 second
        self.robots_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache robots.txt for 1 hour
        self.url_fingerprints = set()
        self.redis_pool = aioredis.ConnectionPool.from_url("redis://localhost", max_connections=10)

    def update_settings(self, new_settings: CrawlSettings):
        self.settings.START_URL = new_settings.start_url
        self.settings.MAX_DEPTH = new_settings.max_depth
        self.settings.CONCURRENT_REQUESTS = new_settings.concurrent_requests
        self.settings.RETRY_LIMIT = new_settings.retry_limit
        self.settings.DELAY_RANGE = new_settings.delay_range
        self.settings.MAX_CRAWL_TIME = new_settings.max_crawl_time

    async def get_redis_connection(self):
        return aioredis.Redis(connection_pool=self.redis_pool)

    def compute_content_fingerprint(self, content: str) -> int:
        return simhash.Simhash(content).value

    def compute_url_fingerprint(self, url: str) -> int:
        """Compute a unique fingerprint for a URL."""
        normalized_url = self.normalize_url(url)
        return hash(normalized_url)

    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        normalized = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            path=parsed.path.rstrip('/') or '/'
        )
        return normalized.geturl()

    async def adjust_concurrency(self, response_time):
        """Dynamically adjust concurrency based on response time."""
        if response_time > 5:  # If response time is greater than 5 seconds
            self.concurrent_requests = max(1, self.concurrent_requests - 10)
            self.rate_limit_delay = min(5, self.rate_limit_delay * 1.5)
        elif response_time < 1:  # If response time is less than 1 second
            self.concurrent_requests = min(settings.CONCURRENT_REQUESTS, self.concurrent_requests + 5)
            self.rate_limit_delay = max(0.1, self.rate_limit_delay * 0.9)

    async def crawl_worker(self, worker_id: int):
        while True:
            url, depth = await self.queue.get()
            try:
                if depth > settings.MAX_DEPTH or url in self.visited:
                    self.uncrawled_links.add(url)
                else:
                    async with self.semaphore:
                        start_time = time.time()
                        await self.crawl_url(url, depth)
                        end_time = time.time()
                        logger.info(f"Worker {worker_id} crawled URL",
                                    url=url,
                                    depth=depth,
                                    time_taken=end_time - start_time)
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered an error", url=url, error=str(e))
                self.handle_error(url, depth, type(e).__name__, str(e))
            finally:
                self.queue.task_done()

    async def crawl_url(self, url: str, depth: int):
        if not self.should_crawl(url):
            return set(), depth + 1

        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

        if robots_url not in self.robots_cache:
            self.robots_cache[robots_url] = await self.fetch_robots_txt(robots_url)

        rp = self.robots_cache[robots_url]
        if not rp.is_allowed("*", url):
            logger.info("URL disallowed by robots.txt", url=url)
            return set(), depth + 1

        crawl_delay = rp.get_crawl_delay("*")
        if crawl_delay:
            await asyncio.sleep(crawl_delay)
        if url in self.visited:
            return set(), depth + 1

        start_time = time.time()

        self.visited.add(url)
        logger.info("Crawling", url=url, depth=depth, links_found=len(self.visited))

        retry_count = 0
        max_retries = 5
        base_timeout = 240000

        while retry_count < max_retries:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context()
                    page = await context.new_page()

                    timeout = base_timeout * (2 ** retry_count)
                    await page.goto(url, wait_until='networkidle', timeout=timeout)

                    html_content = await page.content()
                    soup = BeautifulSoup(html_content, "html.parser")
                    links = await extract_links_from_soup(soup, url)
                    logger.info(f"Extracted {len(links)} links from {url}")

                    for link in links:
                        if link not in self.visited:
                            logger.info(f"Adding link to queue: {link}")
                            await self.queue.put((link, depth + 1))

                    try:
                        content = extract_content_from_soup(soup, url, strategy='advanced')

                        # Ensure content is a dictionary
                        if not isinstance(content, dict):
                            raise ValueError(f"Invalid content type: {type(content)}")

                        # Ensure processedContent exists and is a list or string
                        processed_content = content.get('processedContent', content.get('content', ''))
                        if isinstance(processed_content, list):
                            processed_content = ' '.join(processed_content)
                        elif not isinstance(processed_content, str):
                            processed_content = str(processed_content)

                        content_fingerprint = self.compute_content_fingerprint(processed_content)

                    except Exception as e:
                        logger.error(f"Error extracting content for {url}: {str(e)}")
                        content = {"title": "Error extracting content", "content": str(e)}
                        processed_content = str(e)
                        content_fingerprint = self.compute_content_fingerprint(str(e))

                    if content_fingerprint in self.content_fingerprints:
                        logger.info("Duplicate content found", url=url)
                        await browser.close()
                        return set(), depth + 1

                    self.content_fingerprints.add(content_fingerprint)

                    crawled_data = {
                        "data": [{
                            "url": url,
                            "title": content.get('title', 'No title'),
                            "content": processed_content,
                            "images": {"data": extract_images(soup, url)},
                            "content_fingerprint": content_fingerprint
                        }],
                        "links": list(await extract_links_from_soup(soup, url))
                    }

                    encoded_data = json.dumps(crawled_data, ensure_ascii=False)
                    redis_key = f"url:{urllib.parse.quote_plus(url)}"
                    await redis_client.set(redis_key, encoded_data)
                    logger.info(f"Data saved to Redis - Key: {redis_key}, Data length: {len(encoded_data)}")

                    # Save the data to Redis
                    redis = await self.get_redis_connection()
                    await redis.set(f"url:{url}", json.dumps(crawled_data, ensure_ascii=False))
                    logger.info(f"Saved data for URL {url} to Redis")

                    await self.batch_queue.put((url, json.dumps(crawled_data)))
                    logger.info(f"Added data for URL {url} to batch queue")
                    await browser.close()

                end_time = time.time()
                response_time = end_time - start_time
                await self.adjust_concurrency(response_time)
                await asyncio.sleep(self.rate_limit_delay)

                if url in self.failed_links:
                    del self.failed_links[url]

                return links, depth + 1

            except (PlaywrightError, PlaywrightTimeoutError) as e:
                retry_count += 1
                error_type = type(e).__name__
                error_message = str(e)
                logger.warning(f"Attempt {retry_count} failed for {url}: {error_type} - {error_message}")

                if "net::ERR_SOCKET_NOT_CONNECTED" in error_message:
                    await asyncio.sleep(self.calculate_backoff(retry_count))
                elif "net::ERR_NAME_NOT_RESOLVED" in error_message:
                    logger.error(f"DNS resolution failed for {url}")
                    break
                else:
                    self.handle_error(url, depth, error_type, error_message)
                    if retry_count < max_retries:
                        await asyncio.sleep(self.calculate_backoff(retry_count))
                    else:
                        break

            except Exception as e:
                self.handle_error(url, depth, type(e).__name__, str(e))
                break

        if retry_count == max_retries:
            logger.error(f"Max retries reached for {url}")
            self.uncrawled_links.add(url)

        return set(), depth + 1

    async def fetch_robots_txt(self, robots_url: str):
        rp = robotexclusionrulesparser.RobotExclusionRulesParser()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                response = await page.goto(robots_url)
                if response.ok:
                    content = await page.content()
                    rp.parse(content)
                await browser.close()
        except Exception as e:
            logger.error(f"Error fetching robots.txt: {str(e)}")
        return rp

    def should_crawl(self, url: str) -> bool:
        url_fingerprint = self.compute_url_fingerprint(url)
        if url_fingerprint in self.url_fingerprints:
            return False
        self.url_fingerprints.add(url_fingerprint)
        return True

    def calculate_backoff(self, retry_count):
        return min(300, (2 ** retry_count) + (random.randint(0, 1000) / 1000))

    def handle_error(self, url: str, depth: int, error_type: str, error_message: str):
        if url not in self.failed_links:
            self.failed_links[url] = {'depth': depth, 'attempts': 1, 'error_type': error_type,
                                      'error_message': error_message}
        else:
            self.failed_links[url]['attempts'] += 1

        if self.failed_links[url]['attempts'] <= settings.RETRY_LIMIT:
            logger.info("Retrying URL", url=url, attempts=self.failed_links[url]['attempts'])
            self.queue.put_nowait((url, depth))
        else:
            logger.error("Max retries reached for URL", url=url, error_type=error_type, error_message=error_message)

    async def auto_scale(self):
        while True:
            queue_size = self.queue.qsize()
            if queue_size > settings.SCALE_UP_THRESHOLD and self.current_instances < settings.MAX_INSTANCES:
                self.current_instances += 1
                logger.info(f"Scaling up to {self.current_instances} instances")
                asyncio.create_task(self.crawl_worker(self.current_instances))
            elif queue_size < settings.SCALE_DOWN_THRESHOLD and self.current_instances > settings.MIN_INSTANCES:
                self.current_instances -= 1
                logger.info(f"Scaling down to {self.current_instances} instances")
            await asyncio.sleep(60)  # Check every minute

    async def batch_writer(self):
        redis = await self.get_redis_connection()
        batch = {}
        last_write_time = time.time()
        while True:
            try:
                url, data = await self.batch_queue.get()
                batch[f"url:{url}"] = data  # Use the "url:" prefix

                current_time = time.time()
                if len(batch) >= settings.BATCH_SIZE or (
                        current_time - last_write_time) > 5 or self.batch_queue.empty():
                    pipe = redis.pipeline()
                    for key, value in batch.items():
                        pipe.set(key, value)
                    await pipe.execute()
                    batch = {}
                    last_write_time = current_time

                if self.batch_queue.empty():
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in batch writer: {str(e)}")
            finally:
                self.batch_queue.task_done()

    async def final_flush(self):
        logger.info("Performing final flush of batch queue")
        redis = await self.get_redis_connection()
        while not self.batch_queue.empty():
            try:
                url, data = await self.batch_queue.get()
                await redis.set(f"url:{url}", data)
                logger.info(f"Final flush: Data for URL {url} written to Redis")
            except Exception as e:
                logger.error(f"Error in final flush for URL {url}: {str(e)}")
            finally:
                self.batch_queue.task_done()
        logger.info("Final flush completed")

    async def verify_redis_data(self):
        logger.info("Verifying Redis data")
        cursor = 0
        keys = set()
        while True:
            cursor, partial_keys = await redis_client.scan(cursor, match="*", count=1000)
            keys.update(partial_keys)
            if cursor == 0:
                break

        logger.info(f"Total keys in Redis: {len(keys)}")

        # Log a sample of keys (up to 5)
        sample_keys = list(keys)[:5]
        for key in sample_keys:
            logger.info(f"Sample key: {key}")
            value = await redis_client.get(key)
            if value:
                logger.info(f"Value for {key}: {value[:100]}...")
            try:
                decoded_value = json.loads(value)
                logger.info(f"Sample data - Key: {key}")
                logger.info(f"Title: {decoded_value['data'][0]['title']}")
                logger.info(f"Content snippet: {decoded_value['data'][0]['content'][:100]}...")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON for key: {key}")
            except Exception as e:
                logger.error(f"Error processing key {key}: {str(e)}")

    async def content_processor(self):
        while True:
            try:
                url, content = await self.processing_queue.get()
                # No heavy processing is done here anymore
                await redis_client.set(f"processed:{url}", json.dumps(content))
            except Exception as e:
                logger.error("Error in content processor", error=str(e))
            finally:
                self.processing_queue.task_done()

    async def flush_batch_queue(self):
        batch = []
        while not self.batch_queue.empty():
            url, data = await self.batch_queue.get()
            batch.append((url, data))

        if batch:
            pipe = redis_client.pipeline()
            for url, data in batch:
                pipe.set(url, data)
            results = await pipe.execute()
            logger.info(f"Final batch of {len(batch)} items written to Redis. Results: {results}")

    async def get_domain_stats(self):
        domain_stats = defaultdict(lambda: {"count": 0, "successful": 0, "failed": 0})
        async for key in redis_client.scan_iter(match="url:*"):
            url = key.split(":", 1)[1]
            domain = urlparse(url).netloc
            domain_stats[domain]["count"] += 1

            # Check if the URL was successfully crawled
            data = await redis_client.get(key)
            if data:
                domain_stats[domain]["successful"] += 1
            else:
                domain_stats[domain]["failed"] += 1

        return dict(domain_stats)

    async def get_performance_metrics(self):
        start_time = await redis_client.get("crawl_start_time")
        end_time = await redis_client.get("crawl_end_time")

        if start_time and end_time:
            total_time = float(end_time) - float(start_time)
            urls_crawled = len(self.visited)
            avg_time_per_url = total_time / urls_crawled if urls_crawled > 0 else 0

            return {
                "total_crawl_time": total_time,
                "urls_crawled": urls_crawled,
                "avg_time_per_url": avg_time_per_url
            }
        else:
            return {"error": "Crawl timing information not available"}

    async def get_error_stats(self):
        error_stats = defaultdict(int)
        for url, data in self.failed_links.items():
            error_stats[data['error_type']] += 1
        return dict(error_stats)

    async def start_crawl(self):
        start_url = settings.START_URL
        await self.queue.put((start_url, 0))

        workers = [asyncio.create_task(self.crawl_worker(i)) for i in range(settings.MIN_INSTANCES)]
        auto_scale_task = asyncio.create_task(self.auto_scale())
        batch_writer_task = asyncio.create_task(self.batch_writer())  # Ensure this is added
        content_processor_task = asyncio.create_task(self.content_processor())

        await self.queue.join()

        for worker in workers:
            worker.cancel()

        auto_scale_task.cancel()
        batch_writer_task.cancel()
        content_processor_task.cancel()

        await asyncio.gather(*workers, auto_scale_task, batch_writer_task, content_processor_task,
                             return_exceptions=True)
        await redis_client.set("crawl_end_time", time.time())

    async def run(self):

        crawl_task = asyncio.create_task(self.start_crawl())
        batch_writer_task = asyncio.create_task(self.batch_writer())
        content_processor_task = asyncio.create_task(self.content_processor())

        await crawl_task

        # Final flush of the batch queue
        await self.final_flush()

        # Cancel background tasks
        batch_writer_task.cancel()
        content_processor_task.cancel()

        # Wait for background tasks to complete
        await asyncio.gather(batch_writer_task, content_processor_task, return_exceptions=True)

        logger.info("All tasks completed, crawl finished")

        # Verify data in Redis
        await self.verify_redis_data()

        # Check for missing URLs
        redis_keys = set()
        cursor = '0'
        while cursor != 0:
            cursor, keys = await redis_client.scan(cursor=cursor, match="*", count=1000)
            redis_keys.update(keys)

        missing_urls = self.visited - redis_keys

        if missing_urls:
            logger.warning(f"Some crawled URLs are missing from Redis: {missing_urls}")
        else:
            logger.info("All crawled URLs are present in Redis")


crawler = Crawler()


@app.get("/domain_stats")
async def domain_stats():
    try:
        stats = await crawler.get_domain_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error fetching domain stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance_metrics")
async def performance_metrics():
    try:
        metrics = await crawler.get_performance_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error fetching performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/error_stats")
async def error_stats():
    try:
        stats = await crawler.get_error_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error fetching error stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crawl_progress")
async def crawl_progress():
    try:
        total_urls = len(crawler.visited) + crawler.queue.qsize()
        progress = (len(crawler.visited) / total_urls) * 100 if total_urls > 0 else 0
        return {
            "total_urls": total_urls,
            "crawled_urls": len(crawler.visited),
            "remaining_urls": crawler.queue.qsize(),
            "progress_percentage": progress
        }
    except Exception as e:
        logger.error(f"Error fetching crawl progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    try:
        await redis_client.ping()
        logger.info("Successfully connected to Redis")
        await redis_client.set("crawl_start_time", time.time())
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        return

    logger.info("Starting Crawler")
    # await crawler.run()
    logger.info("Crawling completed")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI web crawler!"}


@app.post("/start_crawl")
async def start_crawl(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(crawler.run)
        return {"message": "Crawl started in the background."}
    except Exception as e:
        logger.error(f"Error starting crawl: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/crawl_results")
async def crawl_results():
    keys = await redis_client.keys("url:*")
    data = {key: await redis_client.hgetall(key) for key in keys}
    return data


@app.get("/health")
async def health_check():
    try:
        await redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "redis": str(e)}


@app.get("/crawl_stats")
async def crawl_stats():
    num_visited = await redis_client.scard("visited_urls")
    num_failed = len(await redis_client.smembers("failed_urls"))
    return {
        "num_visited": num_visited,
        "num_failed": num_failed,
        "visited_urls": await redis_client.smembers("visited_urls"),
        "failed_urls": await redis_client.smembers("failed_urls"),
        "visited_domains": await redis_client.smembers("visited_domains"),
        "current_instances": crawler.current_instances,
        "queue_size": crawler.queue.qsize(),
        "batch_queue_size": crawler.batch_queue.qsize(),
        "processing_queue_size": crawler.processing_queue.qsize()
    }


@app.get("/content_stats")
async def content_stats():
    return {
        "total_urls_crawled": len(crawler.visited),
        "unique_content_count": len(crawler.content_fingerprints),
        "duplicate_content_count": len(crawler.visited) - len(crawler.content_fingerprints)
    }


@app.get("/failed_attempts")
async def failed_attempts():
    return crawler.failed_links


@app.get("/url_data/{url}")
async def get_url_data(url: str):
    encoded_url = urllib.parse.quote_plus(url)
    data = await redis_client.get(f"url:{encoded_url}")
    if data:
        return json.loads(data)
    return {"error": "URL not found"}





@app.get("/queue_status")
async def queue_status():
    return {
        "queue_size": crawler.queue.qsize(),
        "batch_queue_size": crawler.batch_queue.qsize(),
        "processing_queue_size": crawler.processing_queue.qsize()
    }


@app.post("/start_custom_crawl")
async def start_custom_crawl(settings: CrawlSettings, background_tasks: BackgroundTasks):
    try:
        # Update crawler settings
        crawler.update_settings(settings)

        # Reset crawler state
        crawler.visited.clear()
        crawler.uncrawled_links.clear()
        crawler.failed_links.clear()
        crawler.content_fingerprints.clear()
        crawler.url_fingerprints.clear()

        # Reinitialize the queue with the new start URL
        crawler.queue = asyncio.Queue()
        await crawler.queue.put((settings.start_url, 0))

        # Start the crawl
        background_tasks.add_task(crawler.run)
        return {"message": "Custom crawl started in the background.", "settings": settings.dict()}
    except Exception as e:
        logger.error(f"Error starting custom crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()
    logger.info("Crawler shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
