FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser binaries and OS dependencies
RUN playwright install --with-deps chromium || true

# Copy application source
COPY . .

# Create persistent directories
RUN mkdir -p models logs browser_profile

# Default entrypoint runs in safe SIGNAL_ONLY mode
CMD ["python", "main.py", "--mode", "signal"]

