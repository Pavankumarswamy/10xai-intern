# AI Intern - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCKER CONTAINER                             │
│                   (ai-intern-3tasks)                            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              GRADIO WEB INTERFACE                         │ │
│  │              http://localhost:7860                        │ │
│  │                                                           │ │
│  │  ┌─────────────┬─────────────────┬─────────────────────┐ │ │
│  │  │   TASK 1    │     TASK 2      │      TASK 3         │ │ │
│  │  │             │                 │                     │ │ │
│  │  │  Text Chat  │  Speech Chat    │  Unified Interface  │ │ │
│  │  │             │                 │                     │ │ │
│  │  │  User Text  │  User Audio     │  Text OR Audio      │ │ │
│  │  │      ↓      │      ↓          │      ↓              │ │ │
│  │  │   Qwen 3    │   Whisper STT   │   Mode Selector     │ │ │
│  │  │      ↓      │      ↓          │      ↓              │ │ │
│  │  │  AI Text    │   Qwen 3 LLM    │   Qwen 3 LLM        │ │ │
│  │  │             │      ↓          │      ↓              │ │ │
│  │  │             │   Edge TTS      │   Edge TTS          │ │ │
│  │  │             │      ↓          │      ↓              │ │ │
│  │  │             │  AI Audio       │  Text + Audio       │ │ │
│  │  └─────────────┴─────────────────┴─────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    AI MODELS & SERVICES                   │ │
│  │                                                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │ │
│  │  │   Ollama     │  │   Whisper    │  │   Edge TTS    │  │ │
│  │  │   Server     │  │   tiny.en    │  │  (Cloud API)  │  │ │
│  │  │              │  │              │  │               │  │ │
│  │  │  Qwen 3 4B   │  │   39M params │  │  Neural Voice │  │ │
│  │  │  4B params   │  │   STT Model  │  │  Synthesis    │  │ │
│  │  └──────────────┘  └──────────────┘  └───────────────┘  │ │
│  │                                                           │ │
│  │  Port: 11434       Loaded on Use    Requires Internet    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  SYSTEM DEPENDENCIES                      │ │
│  │                                                           │ │
│  │  • Python 3.10        • PyTorch          • FFmpeg        │ │
│  │  • Gradio             • Torchaudio       • Soundfile     │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Exposed Port: 7860
                              ↓
                  http://localhost:7860
```

## Data Flow

### Task 1: Text-to-Text
```
User Browser → Gradio UI → Qwen 3 LLM → Response → Gradio UI → User Browser
```

### Task 2: Speech-to-Speech
```
User Browser → Audio Upload → Whisper (STT) → Text
                                                  ↓
                                              Qwen 3 LLM
                                                  ↓
                                              Edge TTS
                                                  ↓
                                              Audio File
                                                  ↓
                                              Gradio UI → User Browser
```

### Task 3: Unified
```
User selects mode → [Text Path OR Speech Path] → Response
```

## Component Responsibilities

| Component | Purpose | Location |
|-----------|---------|----------|
| Gradio | Web UI & routing | Container |
| Ollama | LLM serving | Container (port 11434) |
| Qwen 3 | Language model | Container (via Ollama) |
| Whisper | Speech-to-text | Container (PyTorch) |
| Edge TTS | Text-to-speech | Cloud (Microsoft) |
| FFmpeg | Audio processing | Container |

## Network Ports

- **7860**: Gradio web interface (exposed to host)
- **11434**: Ollama API (internal only)

## Storage

- **Models**: Stored in container at `/root/.ollama/models/`
- **Whisper Cache**: Downloaded to container on first use
- **Temp Audio**: Created in `/tmp/`, auto-cleaned

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Text Chat | 1-2s | CPU inference |
| Speech Transcription | 500ms | Whisper tiny.en |
| TTS Generation | 500ms | Cloud API |
| Full Speech Pipeline | 2-3s | Combined latency |

## Security Considerations

- ✅ LLM runs locally (no data sent to cloud)
- ✅ STT runs locally (Whisper)
- ⚠️ TTS uses cloud API (Edge TTS)
- ✅ No persistent data storage
- ✅ Temporary files auto-cleaned

## Scalability

**Current Setup:**
- Single container
- CPU inference
- Suitable for demo/development

**Production Considerations:**
- Add GPU support for faster inference
- Load balancing for multiple users
- Model caching strategies
- Replace Edge TTS with local alternative

---

**Architecture Version**: 1.0  
**Last Updated**: January 2026
