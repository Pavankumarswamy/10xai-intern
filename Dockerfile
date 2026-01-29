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

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose Gradio port
EXPOSE 7860

# Create startup script to run Ollama and App
RUN echo '#!/bin/bash\n\
ollama serve & \n\
OLLAMA_PID=$! \n\
sleep 5 \n\
echo "Pulling Gemma 3 model..." \n\
ollama pull gemma3:1b \n\
echo "Starting App..." \n\
python app.py \n\
' > /app/start.sh && chmod +x /app/start.sh

# Run
CMD ["/app/start.sh"]
