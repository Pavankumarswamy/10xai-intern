FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create directory for Ollama models and set permissions
RUN mkdir -p /.ollama && chmod 777 /.ollama
ENV OLLAMA_MODELS=/.ollama

# Give permissions to the app directory
RUN chown -R user:user /app

# Switch to the "user" user
USER user

# Pre-download the LLM model
# We need to run ollama serve in background, wait, pull, then kill it
RUN ollama serve & \
    sleep 5 && \
    ollama pull gemma3:1b

# Pre-install Heavy Python Dependencies (Cached Layer)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir openai-whisper sentence-transformers

# Install Python requirements
# Add local bin to PATH for pip installs
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user app.py .
COPY --chown=user school_information.pdf .

# Expose the port
EXPOSE 7860

# Create start script
RUN echo '#!/bin/bash\n\
ollama serve & \n\
sleep 5 \n\
echo "Starting Unified AI Agent..." \n\
python app.py \n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
