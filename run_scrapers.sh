#!/bin/bash

# Script to run all AI model scrapers using Poetry in WSL
# Scheduled to run daily at 10 AM via cron

# Load user environment for cron
source ~/.bashrc 2>/dev/null || true
source ~/.profile 2>/dev/null || true

# Add common paths where Poetry might be installed
export PATH="$HOME/.local/bin:$PATH:/usr/local/bin"

# Configuration
PROJECT_DIR="/mnt/d/finalms/cost_optimization"
LOG_DIR="$HOME/scraper_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/scraper_run_$TIMESTAMP.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Logging functions
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_to_syslog() {
    logger -t "scraper_cron" "$1"
    log_with_timestamp "$1"
}

# Notifications
notify_windows() {
    local message="$1"
    powershell.exe -Command "Import-Module BurntToast; New-BurntToastNotification -Text 'AI Scrapers', '$message'" 2>/dev/null
}

notify_user() {
    local message="$1"
    log_to_syslog "$message"

    if [[ -n "$DISPLAY" ]] && command -v notify-send &> /dev/null; then
        notify-send "AI Scrapers" "$message" 2>/dev/null || true
    fi

    notify_windows "$message"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" >> "$HOME/scraper_status.log"
}

# Redirect stderr to log file
exec 2>>"$LOG_FILE"

notify_user " DÉMARRAGE: Scraping des modèles IA en cours..."

# Move to project directory
if ! cd "$PROJECT_DIR"; then
    log_with_timestamp "ERROR: Could not navigate to project directory $PROJECT_DIR"
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    log_with_timestamp "ERROR: Poetry is not installed or not in PATH"
    exit 1
fi

# Load environment variables if .env exists
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
    log_with_timestamp "Environment variables loaded from .env"
else
    log_with_timestamp "WARNING: .env file not found at $PROJECT_DIR/.env"
fi

# Ensure Poetry environment is ready
if ! poetry env info > /dev/null 2>&1; then
    log_with_timestamp "Poetry environment not set up. Running 'poetry install'..."
    poetry install >> "$LOG_FILE" 2>&1
fi

# Function to run a scraper with consistent logging
run_scraper() {
    local script_name="$1"
    notify_user " Exécution de $script_name..."
    if poetry run python "$script_name" >> "$LOG_FILE" 2>&1; then
        notify_user " $script_name terminé avec succès"
    else
        local code=$?
        notify_user " ERREUR: $script_name a échoué (code: $code)"
    fi
    sleep 5
}

# Run all scrapers
run_scraper open_llm_scraper.py
run_scraper ai_model_crawler.py
run_scraper anthropicandcoherescraper.py

notify_user " TERMINÉ: Scraping des modèles IA complété avec succès !"

# Clean up old log files (keep last 50 only)
find "$LOG_DIR" -name "scraper_run_*.log" -type f | sort | head -n -50 | xargs -r rm

echo "Run completed. Check $LOG_FILE for details."
