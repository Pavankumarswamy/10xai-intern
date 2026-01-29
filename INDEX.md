# 📚 AI Intern - Documentation Index

Welcome to the AI Intern project! This folder contains everything you need to run 3 AI tasks in a single Docker container.

## 🚀 Start Here

**New to this project?** → Read [`QUICKSTART.md`](QUICKSTART.md)

**Want detailed instructions?** → Read [`README.md`](README.md)

**Curious about the architecture?** → Read [`ARCHITECTURE.md`](ARCHITECTURE.md)

## 📁 File Guide

### 🎯 Essential Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `run-all-projects.ps1` | **Main script** - Build & run everything | **START HERE** |
| `app.py` | Application code (3 tasks) | View/modify code |
| `Dockerfile` | Container configuration | Customize build |
| `requirements.txt` | Python dependencies | Add packages |

### 📖 Documentation

| File | Content |
|------|---------|
| `QUICKSTART.md` | Quick reference (5 min read) |
| `README.md` | Complete guide (15 min read) |
| `ARCHITECTURE.md` | Technical details & diagrams |
| `INDEX.md` | This file |

### 🛠️ Scripts

| Script | Purpose |
|--------|---------|
| `run-all-projects.ps1` | Build + Run (recommended) |
| `build-docker.ps1` | Build image only |
| `run-docker.ps1` | Run existing image |
| `docker-commands.sh` | Linux/Mac commands |

### ⚙️ Configuration

| File | Purpose |
|------|---------|
| `.dockerignore` | Exclude files from image |

## 🎯 The 3 Tasks

### Task 1: Text-to-Text Chat
- Interactive AI conversation
- Powered by Qwen 3 (4B)
- Maintains conversation history

### Task 2: Speech-to-Speech
- Voice input → AI voice output
- Uses Whisper (STT) + Qwen 3 + Edge TTS
- Full conversation log

### Task 3: Unified Interface
- Single interface for both modes
- Mode selector (Text/Speech)
- Flexible input options

## 🏃 Quick Commands

```powershell
# Build and run everything
.\run-all-projects.ps1

# Just build
.\build-docker.ps1

# Just run (after building)
.\run-docker.ps1

# Stop container
# Press Ctrl+C in the running terminal
```

## 🌐 Access Points

Once running:
- **Web Interface**: http://localhost:7860
- **Ollama API**: http://localhost:11434 (internal)

## 📊 Project Stats

- **Total Files**: 11
- **Lines of Code**: ~400 (app.py)
- **Docker Image Size**: ~4GB (includes models)
- **Build Time**: 5-10 minutes (first time)
- **Run Time**: ~30 seconds (after build)

## 🔄 Typical Workflow

1. **First Time Setup**:
   ```powershell
   .\run-all-projects.ps1
   ```
   - Builds image (~10 min)
   - Starts container
   - Opens at http://localhost:7860

2. **Daily Use**:
   ```powershell
   .\run-docker.ps1
   ```
   - Uses existing image
   - Starts in ~30 seconds

3. **After Code Changes**:
   ```powershell
   .\build-docker.ps1
   .\run-docker.ps1
   ```

## 🎓 Learning Path

**Beginner:**
1. Read `QUICKSTART.md`
2. Run `run-all-projects.ps1`
3. Try all 3 tasks in the web UI

**Intermediate:**
1. Read `README.md`
2. Explore `app.py` code
3. Modify prompts or UI

**Advanced:**
1. Read `ARCHITECTURE.md`
2. Customize `Dockerfile`
3. Add new models or features

## 🆘 Help & Troubleshooting

**Issue** | **Solution**
----------|-------------
Docker not running | Start Docker Desktop
Build fails | Check internet, disk space
Port in use | Stop other containers
Slow performance | Ensure 8GB+ RAM available

See `README.md` → Troubleshooting section for details.

## 📦 What's Included?

- ✅ Qwen 3 (4B) language model
- ✅ Whisper speech recognition
- ✅ Edge TTS text-to-speech
- ✅ Gradio web interface
- ✅ Complete documentation
- ✅ Ready-to-run scripts

## 🎯 Success Criteria

You'll know it's working when:
- ✅ Container builds without errors
- ✅ Browser opens to http://localhost:7860
- ✅ You see 3 tabs (Task 1, 2, 3)
- ✅ Text chat responds to messages
- ✅ Speech chat transcribes and responds

## 📞 Next Steps

1. **Run the project**: `.\run-all-projects.ps1`
2. **Test all 3 tasks** in the web interface
3. **Read the documentation** to understand how it works
4. **Experiment** with different inputs

---

**Project Status**: ✅ Ready to Run  
**Documentation**: Complete  
**Last Updated**: January 2026

**Happy coding! 🚀**
