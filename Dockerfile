# Multi-stage Dockerfile for Python Full-Stack App
# Stage 1: Base Python image with dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Backend service
FROM base as backend

# Copy shared modules and backend code
COPY shared/ ./shared/
COPY backend/ ./backend/
COPY data/ ./data/
COPY config.yaml ./

# Create necessary directories
RUN mkdir -p output models

# Expose backend port
EXPOSE 8000

# Health check for backend
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run backend
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Frontend service
FROM base as frontend

# Copy shared modules and frontend code
COPY shared/ ./shared/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY config.yaml ./

# Create necessary directories
RUN mkdir -p output

# Expose frontend port
EXPOSE 8501

# Health check for frontend
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run frontend
CMD ["streamlit", "run", "frontend/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"] 