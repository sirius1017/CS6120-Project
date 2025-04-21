#!/bin/bash

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    echo "Created .env file. Please edit it to add your Google API key."
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t recipe-generator .

# Run the container
echo "Starting container..."
docker run --gpus all -it \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/chroma_title:/app/chroma_title \
    -v $(pwd)/chroma_ingredients:/app/chroma_ingredients \
    -v $(pwd)/hf_cache:/app/hf_cache \
    --env-file .env \
    recipe-generator 