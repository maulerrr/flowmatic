# Use the official Python 3.11 slim image as base
FROM python:3.11-slim

# Update system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent Python from writing .pyc files and to buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any) and copy requirements.txt
# psycopg2-binary bundles libpq, so no need for libpq-dev here.
# If you add other libraries that need system headers, install them here.
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8000 (the port Uvicorn will run on)
EXPOSE 8000

# Specify default command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "flowmatic.server:app", "--host", "0.0.0.0", "--port", "8000"]
