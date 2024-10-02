# Web Crawler

## Overview
This project is an advanced web crawler built with Python, leveraging FastAPI for the API layer and Redis for data storage. It's designed to efficiently crawl websites, extract content, and provide various statistics and insights about the crawling process.

## Features
- Asynchronous crawling with dynamic concurrency adjustment
- Content extraction and fingerprinting to avoid duplicates
- Respect for robots.txt
- Auto-scaling of crawler instances
- Batch processing and writing to Redis
- Comprehensive error handling and retrying mechanism
- API endpoints for crawl control and statistics

## Requirements
- Python 3.7+
- FastAPI
- Redis
- Playwright
- BeautifulSoup4
- Pandas
- Spacy
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```
   git clone <[repository-url](https://github.com/Saadmomin2903/Crawler)>
   cd advanced-web-crawler
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```
   playwright install
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add:
   ```
   UPSTASH_REDIS_URL=your_redis_url
   UPSTASH_REDIS_TOKEN=your_redis_token
   START_URL=https://www.example.com
   ```

## Usage
1. Start the FastAPI server:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Use the API endpoints to control the crawler and retrieve statistics:
   - POST `/start_crawl`: Start the crawler
   - GET `/crawl_progress`: Get current crawl progress
   - GET `/domain_stats`: Get statistics per domain
   - GET `/performance_metrics`: Get overall performance metrics
   - GET `/error_stats`: Get error statistics
   - GET `/crawl_results`: Retrieve crawled data
   - GET `/health`: Check the health of the application

## Configuration
You can adjust the crawler's behavior by modifying the `Settings` class in the code. Key settings include:
- `MAX_DEPTH`: Maximum depth for crawling
- `CONCURRENT_REQUESTS`: Number of concurrent requests
- `RETRY_LIMIT`: Number of retry attempts for failed requests
- `DELAY_RANGE`: Range for random delay between requests
- `MAX_CRAWL_TIME`: Maximum time for the entire crawl process

## Advanced Features
- **Content Fingerprinting**: Avoids storing duplicate content using SimHash
- **Auto-scaling**: Dynamically adjusts the number of crawler instances based on queue size
- **Batch Processing**: Efficiently writes data to Redis in batches
- **Custom Crawls**: Allows starting crawls with custom settings via the `/start_custom_crawl` endpoint

## Monitoring and Debugging
- Use the `/crawl_stats` endpoint to get detailed statistics about the crawl process
- Check `/content_stats` for information about unique and duplicate content
- View `/failed_attempts` to see which URLs failed to crawl and why

## Contributing
Contributions to improve the crawler are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License


## Disclaimer
Ensure you have permission to crawl websites and always respect robots.txt rules and the website's terms of service.
