# Use Python slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
RUN touch README.md

# Install dependencies with pip
RUN pip install --upgrade pip && \
    pip install .

# Copy application code
COPY rag_chat_app/ rag_chat_app/
COPY run.py .

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["python", "run.py"]