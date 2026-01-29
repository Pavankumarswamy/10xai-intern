FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies (Cached)
COPY requirements.txt .
# Use BuildKit cache mount to speed up repeated installs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Pre-download the LLM model during build to speed up startup
# We start the server, pull the model to /root/.ollama, and then it is saved in the image layer.
RUN ollama serve & \
    sleep 5 && \
    ollama pull gemma3:1b

# Copy application code and assets
COPY app.py .
COPY school_information.pdf .

# Expose Gradio port
EXPOSE 7860

# Startup script (Model is already baked in)
RUN echo '#!/bin/bash\n\
ollama serve & \n\
sleep 5 \n\
echo "Starting Unified AI Agent..." \n\
python app.py \n\
' > /app/start.sh && chmod +x /app/start.sh

# Run
CMD ["/app/start.sh"]
