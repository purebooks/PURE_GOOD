# Use official Python runtime as base image
FROM python:3.11-slim

# Set environment variables for Cloud Run optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV FLASK_ENV=production
ENV WORKERS=2
ENV THREADS=4
ENV TIMEOUT=300

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application with gunicorn optimized for Cloud Run
# Uses environment variables for flexible scaling
CMD exec gunicorn --bind :$PORT --workers $WORKERS --threads $THREADS --timeout $TIMEOUT --worker-class sync --preload app_v5:app 