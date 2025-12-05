FROM python:3.12-slim

WORKDIR /app

# Install git for installing dependencies from git repositories
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 1. Install CPU-only PyTorch and JAX explicitly first
# We do this BEFORE requirements.txt to ensure these specific versions are used
# and to prevent pip from pulling the massive GPU versions from PyPI later.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "jax[cpu]"

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
