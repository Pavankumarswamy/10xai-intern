---
title: Unified AI Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Unified AI Assistant

This Space demonstrates a unified AI agent performing three tasks:
1. **Text & Speech**: Using Gemma 3:1b and Whisper/EdgeTTS.
2. **LUCA Assistant**: Identity-enforced voice assistant.
3. **School RAG**: Question answering on school documents.

## Running Locally

```bash
docker build -t ai-assistant .
docker run -p 7860:7860 ai-assistant
```
