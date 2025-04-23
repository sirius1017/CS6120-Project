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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/chroma_title /app/chroma_ingredients /app/chroma_instructions /app/hf_cache

# Create a script to download and process data
RUN echo '#!/bin/bash\n\
\n\
# Check if .env file exists\n\
if [ ! -f .env ]; then\n\
    echo "Error: .env file not found. Please create one with your GOOGLE_API_KEY."\n\
    exit 1\n\
fi\n\
\n\
# Check if recipes.json exists\n\
if [ ! -f recipes.json ]; then\n\
    echo "Downloading and processing recipe dataset..."\n\
    python3 download_data.py\n\
else\n\
    echo "Recipe dataset already exists."\n\
fi\n\
\n\
# Start Ollama in the background\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
\n\
# Wait for Ollama to start\n\
sleep 5\n\
\n\
# Pull the Gemma model if not already pulled\n\
echo "Pulling Gemma model..."\n\
ollama pull ${MODEL}\n\
\n\
echo "Starting the recipe generation system..."\n\
streamlit run --server.port=8501 --server.address=0.0.0.0 app.py' > /app/start.sh

# Make the script executable
RUN chmod +x /app/start.sh

# Expose Streamlit port
EXPOSE 8501

# Set the entry point
ENTRYPOINT ["/app/start.sh"] 