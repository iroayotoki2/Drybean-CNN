# Use an official Python image
FROM python:3.10-slim

# Prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (expand as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (improves Docker cache usage)
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Install PLINK 1.9
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20230116.zip \
    -O /tmp/plink.zip && \
    unzip /tmp/plink.zip -d /usr/local/bin && \
    chmod +x /usr/local/bin/plink && \
    rm /tmp/plink.zip

# Copy the rest of the project
COPY . .

