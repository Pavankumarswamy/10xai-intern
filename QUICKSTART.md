# Quick Start Guide - AI Intern Docker

## 🚀 Fastest Way to Run

1. **Make sure Docker Desktop is running**
2. **Open PowerShell in this folder**
3. **Run:**
   ```powershell
   .\run-all-projects.ps1
   ```
4. **Wait for build to complete** (~5-10 minutes first time)
5. **Open browser:** http://localhost:7860

## 📱 What You'll See

Three tabs in the web interface:
- **Task 1**: Text chat with AI
- **Task 2**: Voice chat (speak and hear responses)
- **Task 3**: Combined interface (switch between text/voice)

## 🛑 To Stop

Press `Ctrl+C` in the PowerShell window

## 📝 Files You Need

- `app.py` - The application code
- `Dockerfile` - Container configuration
- `requirements.txt` - Python packages
- `run-all-projects.ps1` - Main script to run

## ❓ Troubleshooting

**Docker not running?**
→ Start Docker Desktop, wait for it to fully load

**Port 7860 in use?**
→ Stop other containers or change port in run script

**Build failed?**
→ Check internet connection, ensure Docker has enough disk space

## 📊 What's Inside the Container?

- Qwen 3 (4B) - AI language model
- Whisper - Speech recognition
- Edge TTS - Text-to-speech
- Gradio - Web interface

---

**Need detailed help?** See `README.md`
