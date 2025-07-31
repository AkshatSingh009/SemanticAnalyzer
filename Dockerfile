# Use a stable Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages and other required libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (adjust if your app runs on a different port)
EXPOSE 8000

# Define default command to run your app (update as per your app entrypoint)
CMD ["python", "main.py"]
