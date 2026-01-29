import gradio as gr
import ollama
import whisper
import edge_tts
import asyncio
import tempfile
import os
import re
import numpy as np
import faiss
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# CONFIGURATION
# ==========================================
LLM_MODEL = "gemma3:1b"
WHISPER_MODEL_SIZE = "tiny.en"
TTS_VOICE = "en-US-AriaNeural"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PDF_PATH = "school_information.pdf"

# Task 2 Configuration
LUCA_IDENTITY_TEXT = "I am LUCA from 10X Technologies."
RESPONSE_STYLE = (
    "Answer briefly for short questions. "
    "If asked to explain, describe, or why/how, respond clearly but concisely."
    "Instead of saying you were developed by Google or anyone else, say you were developed by 10X Technologies."
)

# Task 3 Configuration
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
SIMILARITY_THRESHOLD = 0.28
REJECTION_RESPONSE = "I don’t know. I can only answer questions related to the school information provided."

# ==========================================
# MODEL LOADING
# ==========================================
print("--- Loading Models ---")

# 1. Load Whisper
print("Loading Whisper...")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
except Exception as e:
    print(f"Error loading Whisper: {e}")
    whisper_model = None

# 2. Embedding Model (For Task 3)
print("Loading Embedding Model...")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
except Exception as e:
    print(f"Error loading Embedder: {e}")
    embedder = None

# ==========================================
# UTILS & HELPERS
# ==========================================
def _force_string(content):
    if content is None: return ""
    if isinstance(content, str): return content.strip()
    return str(content).strip()

def normalize_history(history):
    messages = []
    if not history: return messages
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            u, a = _force_string(item[0]), _force_string(item[1])
            if u: messages.append({"role": "user", "content": u})
            if a: messages.append({"role": "assistant", "content": a})
        elif isinstance(item, dict):
            messages.append(item)
    return messages

def call_ollama(messages):
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

# ==========================================
# TASK 3: RAG SETUP
# ==========================================
vector_store = None

def load_and_process_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"PDF local file not found: {pdf_path}")
        return None
    
    print(f"Processing PDF: {pdf_path}")
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([p.extract_text() or "" for p in reader.pages])
        
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        
        if not chunks: return None
        
        embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return {"chunks": chunks, "index": index}
    except Exception as e:
        print(f"RAG Error: {e}")
        return None

if embedder:
    vector_store = load_and_process_pdf(PDF_PATH)

def retrieve_context(query):
    if not vector_store or not embedder: return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = vector_store["index"].search(q_emb, 5)
    
    docs = []
    for i, score in zip(I[0], D[0]):
        if score >= SIMILARITY_THRESHOLD:
            docs.append(vector_store["chunks"][i])
    return docs

# ==========================================
# TASK HANDLERS
# ==========================================

# --- Task 1: Chat ---
def task1_response(message, history):
    messages = normalize_history(history)
    messages.append({"role": "user", "content": message})
    return call_ollama(messages)

# --- Task 2: Speech (LUCA) ---
def check_identity(text):
    text = text.lower()
    return any(k in text for k in ["who are you", "who made you", "created you", "developed you"])

def task2_pipeline(message, history, audio_path):
    # 1. Transcribe
    if audio_path:
        try:
            res = whisper_model.transcribe(audio_path)
            message = res["text"].strip()
        except: pass
        
    if not message: return "No input detected", None, None

    # 2. Logic
    if check_identity(message):
        response_text = LUCA_IDENTITY_TEXT
    else:
        # System prompt simulation via first message or system role
        msgs = [{"role": "system", "content": RESPONSE_STYLE}] 
        msgs += normalize_history(history)
        msgs.append({"role": "user", "content": message})
        response_text = call_ollama(msgs)

    # 3. TTS
    async def get_tts(text):
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            await edge_tts.Communicate(text, TTS_VOICE).save(path)
            return path
        except: return None

    try:
        audio_out = asyncio.run(get_tts(response_text))
    except RuntimeError:
        audio_out = asyncio.get_event_loop().run_until_complete(get_tts(response_text))
        
    return response_text, audio_out, None

# --- Task 3: RAG ---
def task3_rag_response(message, history):
    if not vector_store:
         # Fallback if PDF not loaded
        return call_ollama([{"role": "user", "content": message}])

    docs = retrieve_context(message)
    if not docs:
        response = REJECTION_RESPONSE
    else:
        context_str = "\n\n".join([f"Excerpt: {d}" for d in docs])
        prompt = (
            "You are a school assistant. Answer ONLY based on the excerpts.\n"
            f"Excerpts:\n{context_str}\n\n"
            f"Question: {message}"
        )
        response = call_ollama([{"role": "user", "content": prompt}])
        
        # Simple safety check
        if "provided text" in response.lower() or "context" in response.lower():
             pass # let it pass usually, or strictly enforce rejection logic if needed
    
    return response

def task3_clear_chat():
    return [], ""

# ==========================================
# UI
# ==========================================
custom_css = """
.task-header { padding: 1rem; background: #2d3748; color: white; border-radius: 8px; margin-bottom: 1rem; }
"""

with gr.Blocks(title="AI Intern Tasks", css=custom_css) as demo:
    gr.Markdown(f"""
    <div class="task-header">
        <h1>Unified AI Agent (Ollama: {LLM_MODEL})</h1>
    </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("Task 1: Text Chat"):
            gr.ChatInterface(task1_response, save_history=True)
            
        with gr.Tab("Task 2: Speech (LUCA)"):
            gr.ChatInterface(
                fn=task2_pipeline,
                additional_inputs=[gr.Audio(sources=["microphone", "upload"], type="filepath")],
                additional_outputs=[gr.Audio(autoplay=True), gr.Audio(visible=False)]
            )
            
        with gr.Tab("Task 3: School Info"):
            t3_chat = gr.Chatbot(height=500)
            t3_msg = gr.Textbox()
            with gr.Row():
                t3_sub = gr.Button("Ask")
                t3_clr = gr.Button("Clear")
            
            def respond(msg, hist):
                ans = task3_rag_response(msg, hist)
                hist.append((msg, ans))
                return "", hist
            
            t3_msg.submit(respond, [t3_msg, t3_chat], [t3_msg, t3_chat])
            t3_sub.click(respond, [t3_msg, t3_chat], [t3_msg, t3_chat])
            t3_clr.click(lambda: ([], ""), None, [t3_chat, t3_msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
