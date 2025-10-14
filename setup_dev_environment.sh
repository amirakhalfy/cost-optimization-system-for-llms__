#!/bin/bash
 
# Variables
PYTHON_VERSION="3.11.0"
KERNEL_NAME="cost_opt"
DEPENDENCIES=(pandas joblib scikit-learn)
DEV_DEPENDENCIES=(pre-commit ipykernel pylint  pytest pytest-cov)
 
# Install Python version using pyenv
pyenv install "$PYTHON_VERSION"
pyenv local "$PYTHON_VERSION"
 
# Upgrade pip and install Poetry
python -m pip install --upgrade pip
pip install poetry
 
# Prompt user for Poetry initialization
read -p "Do you want to initialize a new Poetry project? (y/n): " init_poetry
 
if [ "$init_poetry" = "y" ]; then
    poetry init
    # Add main dependencies
    for dep in "${DEPENDENCIES[@]}"; do
        poetry add "$dep"
    done
 
    # Add development dependencies
    for dev_dep in "${DEV_DEPENDENCIES[@]}"; do
        poetry add "$dev_dep" --group dev
    done
elif [ "$init_poetry" = "n" ]; then
    poetry install
fi
 
# Add Jupyter kernel
poetry run python -m ipykernel install --user --name "$KERNEL_NAME"
 
# Configure Poetry to use in-project virtual environment
poetry config virtualenvs.in-project true
poetry config virtualenvs.in-project