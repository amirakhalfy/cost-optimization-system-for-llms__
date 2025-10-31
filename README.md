# Cost Optimization System for LLMs

A full-stack system for scraping AI model data, storing and normalizing it in a relational database, serving it via an API, and visualizing usage, budgets, and optimizations with a Streamlit dashboard. It combines daily automation, multi-level caching, cost analysis, and interactive budget forecasting.
## Note: For best results, it is recommended to scrape the websites one by one, with a delay between each request, in order to avoid hitting rate limits

## Table of Contents
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Setup Guide](#setup-guide)
- [Running the Application](#running-the-application)
- [Docker Support](#docker-support)
- [Makefile Shortcuts](#makefile-shortcuts)
- [Data Sources](#data-sources)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Contributing](#contributing)

## Project Structure

### Scraper (WSL or Linux Environment)
- **Scripts**:
  - `open_llm_scraper.py`: Scrapes data from Hugging Face and other open sources.
  - `ai_model_crawler.py`: Uses LLM-based strategy for deep structured scraping.
  - `anthropicandcoherescraper.py`: Scrapes closed-source providers like Anthropic and Cohere.
- **Orchestration**: `run_scrapers.sh` executes all scrapers.
- **Database Integration**: `savingdb.py` normalizes, deduplicates, and persists scraped data.

### FastAPI Backend
- **Entry Point**: `app/main.py`
- **Modules**:
  - `app/sql/`: CRUD endpoints for database tables.
  - `app/aggregation/`: Aggregated comparison and analysis.
  - `app/advancedaggregation/`: Cost-efficiency and open-source filters.
  - `app/caching/`: Multi-layer caching logic.
  - `app/db/`: SQLAlchemy models and connection setup.

### Streamlit Frontend
- Visual interface for exploring model pricing, usage scenarios, budget forecasting, and model invocation.

## Key Features
- Scraping from open and closed model providers.
- LLM-based structured data extraction with retry logic.
- MySQL-backed normalization and historical tracking.
- FastAPI serving CRUD and aggregation endpoints.
- Multi-layer caching (Local, Redis, DB) with TTL.
- Daily automation via cron.
- Streamlit UI for budgeting, analytics, and forecasting.

## Requirements

### System
- **Operating System**: WSL or Linux
- **Python**: ≥ 3.9 (recommended: 3.11.0)
- **Database**: MySQL server
- **Optional**:
  - Redis (for caching)
  - `libnotify` (Linux) or `BurntToast` (Windows) for notifications
- **Tools**:
  - `pyenv` for Python version management
  - `Poetry` for dependency management

### Python Dependencies

#### API + Scraper
```ini
fastapi==0.115.0
uvicorn==0.32.0
sqlalchemy==2.0.35
pydantic==2.9.2
redis==5.1.1
cachetools==5.5.0
httpcore==1.0.5
python-dotenv==1.0.1
mysql-connector-python==9.1.0
streamlit==1.39.0
pandas==2.2.3
numpy==2.1.1
plotly==5.24.1
requests==2.32.3
streamlit-option-menu==0.4.0
```

## Setup Guide

### 1. Clone the Repository
```bash
git clone -b dev --single-branch https://gitlab.com/amira.khalfi/cost_optimization_system-for_llms.git
cd cost_optimization
```

### 2. Python Environment Setup
```bash
pyenv local 3.11.0
poetry env use $(pyenv which python)
source .venv/bin/activate
python --version  # Should output Python 3.11.0
poetry install
```

### 3. Environment Configuration
Create a `.env` file with:
```bash
cat <<EOF > .env
GROQ_API_TOKEN=your-token
CRAWLER_PROXY=optional-proxy
DATABASE_URL=mysql+mysqlconnector://user:pass@localhost:3306/cost_optimization_db
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
EOF
```

### 4. Database Setup
```bash
mysql -u root -p -e "CREATE DATABASE cost_optimization_db;"
poetry run python -m app.db.db_setup

make alembic-init
make alembic-revision-dev msg="Initial database schema"
make alembic-run-revision-dev
```

### 5. Script Configuration
```bash
chmod +x run_scrapers.sh
```

### 6. Cron Job Setup
```bash
crontab -e
```
Add the following line to run daily at 10 AM:
```
0 10 * * * /bin/bash /absolute/path/to/run_scrapers.sh >> ~/scraper_cron.log 2>&1
```

### Notification Setup (Optional)

**Linux:**
```bash
sudo apt-get install libnotify-bin
```

**Windows (PowerShell as Admin):**
```powershell
Install-Module -Name BurntToast
```

## Running the Application

### Scraping
Run individual scrapers:
```bash
poetry run python open_llm_scraper.py
poetry run python ai_model_crawler.py
poetry run python anthropicandcoherescraper.py
```

Run all scrapers:
```bash
./run_scrapers.sh
```

### FastAPI Backend
```bash
make start_dev  # With auto-reload
make fastapi    # Without reload
```

### Streamlit Frontend
```bash
make streamlit
```

## Docker Support
Run the database (and optionally Redis) using Docker:
```bash
docker-compose up -d
```
Note: Assumes a `docker-compose.yml` file defining the database and optional Redis services.

## Makefile Shortcuts

| Command | Description |
|---------|-------------|
| `make start_dev` | Launch FastAPI in dev mode (reload) |
| `make fastapi` | Run FastAPI normally |
| `make streamlit` | Run the Streamlit dashboard |
| `make alembic-init` | Initialize Alembic migrations |
| `make alembic-revision-dev` | Create migration from models |
| `make alembic-run-revision-dev` | Apply migrations to DB |

## Data Sources
- **Hugging Face**: Open-source models
- **Vellum AI**: Benchmark leaderboard
- **OpenAI**: Pricing and tiers
- **Anthropic**: Claude pricing
- **Cohere**: Command pricing
- **Groq**: Performance and cost

### Data Processing
- **Normalization**: Ensures consistent format across providers.
- **Deduplication**: Removes redundant entries.
- **Validation**: Ensures all required fields are present.

## Testing

Run all tests:
```bash
poetry run pytest
```

Run a specific test file:
```bash
poetry run pytest tests/test_api.py -v
```

## Monitoring

### Cron Jobs
List cron jobs:
```bash
crontab -l
```

Check cron service status:
```bash
sudo systemctl status cron
```

Monitor logs:
```bash
tail -f /var/log/cron.log
tail -f ~/scraper_cron.log
```

## Contributing

1. Fork the repository.

2. Create a branch:
   ```bash
   git checkout -b feature/your-feature
   ```

3. Commit changes:
   ```bash
   git commit -m "Add your feature"
   ```

4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```

5. Create a Pull Request.
