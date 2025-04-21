# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a script to download and process data
RUN echo '#!/bin/bash\n\
echo "Downloading and processing recipe dataset..."\n\
python3 download_data.py\n\
echo "Starting the recipe generation system..."\n\
python3 app.py' > /app/start.sh

# Make the script executable
RUN chmod +x /app/start.sh

# Set the entry point
ENTRYPOINT ["/app/start.sh"] 