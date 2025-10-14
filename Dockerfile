###############################################
# Base Image
###############################################
FROM python:3.11-slim as python-base

# Environment variables for Python and Poetry setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.1.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/code" \
    VENV_PATH="/code/.venv"

# Prepend Poetry and virtual environment to the PATH
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

###############################################
# Builder Image
###############################################
FROM python-base as builder-base

# Install build dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev

# Install Poetry (using the official Poetry install script)
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the working directory and copy project files
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml alembic.ini ./

# Install the project dependencies (without dev dependencies)
RUN poetry install --without dev


###############################################
# Production Image
###############################################
FROM python-base as production

# Install curl (for downloading external resources, if needed)
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl

# Set the working directory for the app
WORKDIR $PYSETUP_PATH

# Copy the installed dependencies from the builder stage
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

# Copy the app, migrations, and api folders
COPY app $PYSETUP_PATH/app
COPY api $PYSETUP_PATH/api

# Expose the FastAPI app port (3206)
EXPOSE 3206

# Command to run Alembic migrations and then start the FastAPI app
CMD ["sh", "-c", "poetry run alembic upgrade head && poetry run uvicorn app.main:app --host=0.0.0.0 --port=3206 --reload"]
