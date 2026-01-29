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
# Note: System prompts are used for style, but identity is enforced via code logic.

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
# UTILS & HELPERS
# ==========================================
def _force_string(content):
    if content is None: 
        return ""
    if isinstance(content, str): 
        return content.strip()
    
    # Handle Gradio's multimodal format (list of dicts)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
               text_parts.append(str(part['text']))
            elif isinstance(part, str):
               text_parts.append(part)
        if text_parts:
            return " ".join(text_parts).strip()
            
    # Handle dict (extract text/content)
    if isinstance(content, dict):
        if 'text' in content: return str(content['text']).strip()
        if 'content' in content: return str(content['content']).strip()
        
    # Fallback
    return str(content).strip()

def normalize_history(history):
    messages = []
    if not history: return messages
    
    print(f"DEBUG: Raw History cleanup: {str(history)[:100]}...") # Debug print
    
    for item in history:
        # Format: [[user, bot], ...]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            u = _force_string(item[0])
            a = _force_string(item[1])
            if u: messages.append({"role": "user", "content": u})
            if a: messages.append({"role": "assistant", "content": a})
            
        # Format: {'role': 'user', 'content': ...}
        elif isinstance(item, dict):
            clean_item = item.copy()
            if 'content' in clean_item:
                clean_item['content'] = _force_string(clean_item['content'])
            messages.append(clean_item)
            
    return messages

# Helper for non-streaming calls (Tasks 2 & 3)
def call_ollama_non_stream(messages):
    try:
        response = ollama.chat(model=LLM_MODEL, messages=messages, stream=False)
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

# --- Task 1: Open Source Models ---
def task1_text_response(message, history):
    messages = normalize_history(history)
    # Add system prompt for Task 1
    messages.insert(0, {"role": "system", "content": "You are a helpful AI assistant."})
    # Ensure message is a clean string
    messages.append({"role": "user", "content": _force_string(message)})
    
    try:
        stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
        partial_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            partial_response += content
            yield partial_response
    except Exception as e:
        yield f"Error: {str(e)}"
    return

# Re-use TTS logic for Task 1 speech demo
def task1_speech_response(audio_path):
    if not audio_path: return None, "No audio provided."
    
    # 1. Pipeline: Speech -> Text
    try:
        res = whisper_model.transcribe(audio_path)
        user_text = res["text"].strip()
    except Exception as e:
        return None, f"Transcription error: {e}"

    # 2. Pipeline: Text -> LLM
    # call_ollama_non_stream expects messages list; ensure content is string
    llm_response = call_ollama_non_stream([{"role": "user", "content": _force_string(user_text)}])

    # 3. Pipeline: LLM -> Speech
    async def get_tts(text):
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            await edge_tts.Communicate(text, TTS_VOICE).save(path)
            return path
        except: return None

    try:
        audio_out = asyncio.run(get_tts(llm_response))
    except RuntimeError:
        audio_out = asyncio.get_event_loop().run_until_complete(get_tts(llm_response))

    return audio_out, f"**User:** {user_text}\n\n**AI:** {llm_response}"


# --- Task 2: LUCA Assistant (Identity Enforced) ---
def is_identity_question(text):
    text = text.lower()
    return any(k in text for k in ["who are you", "who made you", "created you", "developed you"])

def task2_luca_handler(message, history, audio_path):
    # 1. Input Processing
    user_input = message
    transcribed_from_audio = False
    
    if audio_path:
        try:
            res = whisper_model.transcribe(audio_path)
            transcribed = res["text"].strip()
            if transcribed:
                user_input = transcribed
                transcribed_from_audio = True
        except Exception as e:
            return f"Transcription error: {str(e)}", None, None
    
    user_input = _force_string(user_input)
    if not user_input:
        return "Please say or type something.", None, None

    # 2. Logic & Identity Check (Hardcoded override)
    if is_identity_question(user_input):
        response_text = LUCA_IDENTITY_TEXT
    else:
        # Standard chat
        msgs = normalize_history(history)
        msgs.append({"role": "user", "content": user_input})
        response_text = call_ollama_non_stream(msgs)

    # 3. Output Processing (TTS)
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
    
    # If transcribed from audio, prepend indicator to response
    if transcribed_from_audio:
        response_text = f"🎤 *Transcribed: \"{user_input}\"*\n\n{response_text}"
        
    return response_text, audio_out, None


# --- Task 3: School RAG ---
def task3_rag_response(message, history):
    if not vector_store:
         # Fallback default behavior
        yield "School PDF not loaded or processed correctly."
        return

    message = _force_string(message)
    docs = retrieve_context(message)
    if not docs:
        yield REJECTION_RESPONSE
        return

    context_str = "\n\n".join([f"Excerpt: {d}" for d in docs])
    prompt = (
        "You are a school assistant provided with the following school information excerpts.\n"
        "Answer the question using strictly ONLY the information from these excerpts.\n"
        "If the answer is not in the excerpts, say 'I don't know'.\n\n"
        f"Excerpts:\n{context_str}\n\n"
        f"Question: {message}"
    )
    
    # Use streaming
    try:
        messages = [{"role": "user", "content": prompt}]
        stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
        partial_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            partial_response += content
            yield partial_response
    except Exception as e:
        yield f"Error: {e}"

def task3_clear_chat():
    return [], ""

# ==========================================
# UI CONSTRUCTION
# ==========================================
custom_css = """
.task-header { padding: 1rem; background: linear-gradient(to right, #2b6cb0, #2c5282); color: white; border-radius: 8px; margin-bottom: 1rem; text-align: center; }
.introduction { font-size: 1.1em; margin-bottom: 1em; }
"""

with gr.Blocks(title="AI Intern Assignments") as demo:
    gr.Markdown(f"""
    <div class="task-header">
        <h1>AI Intern - Unified Task Submission</h1>
        <p>Select a Task from the Tabs below to execute.</p>
    </div>
    """)
    
    with gr.Tabs():
        # TASK 1
        with gr.Tab("Task 1: Open Source Models"):
            gr.Markdown("### Deploy Text & Speech Models")
            gr.Markdown("Objective: Demonstrate 1 Text-to-Text model and 1 Speech-to-Speech model.")
            
            with gr.Accordion("Sub-Task A: Text-to-Text Model (Gemma 3)", open=True):
                gr.ChatInterface(task1_text_response, title="Text Model Demo")
            
            with gr.Accordion("Sub-Task B: Speech-to-Speech Model (Whisper + EdgeTTS)", open=False):
                with gr.Row():
                    t1_audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
                    t1_audio_out = gr.Audio(label="Output Audio", autoplay=True)
                t1_transcript = gr.Markdown(label="Transcript")
                t1_btn = gr.Button("Run Speech Pipeline")
                
                t1_btn.click(task1_speech_response, t1_audio_in, [t1_audio_out, t1_transcript])

        # TASK 2
        with gr.Tab("Task 2: LUCA Voice Assistant"):
            gr.Markdown("### LUCA Voice Assistant")
            gr.Markdown("Objective: Voice-enabled assistant identifying as 'LUCA from 10X Technologies'. Supports both Text & Voice input.")
            
            gr.ChatInterface(
                fn=task2_luca_handler,
                additional_inputs=[
                    gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice Input (Optional)")
                ],
                additional_outputs=[
                    gr.Audio(label="LUCA Voice Reply", autoplay=True),
                    gr.Audio(visible=False) # Clear input
                ],
                title="Talk to LUCA",
                description="Ask 'Who are you?' to verify identity."
            )

        # TASK 3
        with gr.Tab("Task 3: School AI (RAG)"):
            gr.Markdown("### RAG-Based School Assistant")
            gr.Markdown("Objective: Answer domain-specific questions from the School PDF. Rejects unrelated queries.")
            
            t3_chat = gr.Chatbot(height=500, label="School Bot")
            t3_msg = gr.Textbox(label="Question", placeholder="Ask about school fees, timings, etc.")
            with gr.Row():
                t3_sub = gr.Button("Ask")
                t3_clr = gr.Button("Clear Chat")
            
            # Generator wrapper for chatbot
            def respond_t3(msg, hist):
                if not msg.strip(): yield "", hist; return
                
                # Append user message first
                hist.append((msg, ""))
                yield "", hist
                
                # Stream response
                for partial in task3_rag_response(msg, hist):
                    hist[-1] = (msg, partial)
                    yield "", hist
            
            t3_msg.submit(respond_t3, [t3_msg, t3_chat], [t3_msg, t3_chat])
            t3_sub.click(respond_t3, [t3_msg, t3_chat], [t3_msg, t3_chat])
            t3_clr.click(lambda: ([], ""), None, [t3_chat, t3_msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
