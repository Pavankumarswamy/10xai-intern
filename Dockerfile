FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install system dependencies & Ollama
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

# 2. Pre-download the LLM model (Cached Layer)
# Placed BEFORE python requirements to prevent re-downloading when requirements change
RUN ollama serve & \
    sleep 5 && \
    ollama pull gemma3:1b

# 3. Install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 4. Copy application code and assets
COPY app.py .
COPY school_information.pdf .

# Expose Gradio port
EXPOSE 7860

# Startup script
RUN echo '#!/bin/bash\n\
ollama serve & \n\
sleep 5 \n\
echo "Starting Unified AI Agent..." \n\
python app.py \n\
' > /app/start.sh && chmod +x /app/start.sh

# Run
CMD ["/app/start.sh"]
