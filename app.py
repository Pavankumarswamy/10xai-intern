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
# TASK 3: RAG SETUP & DYNAMIC LOADING
# ==========================================
def create_vector_store(pdf_path):
    """Creates a vector store from a PDF file."""
    if not os.path.exists(pdf_path):
        print(f"PDF local file not found: {pdf_path}")
        return None, None
    
    print(f"Processing PDF: {pdf_path}")
    try:
        # Check size for uploaded files (skip check for default if needed, or handle in UI)
        file_size = os.path.getsize(pdf_path)
        if file_size > 2 * 1024 * 1024:  # 2MB limit
             # For default file, we might ignore this, but for upload strict check
             print("Warning: File larger than 2MB")
             pass 

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join([p.extract_text() or "" for p in reader.pages])
        
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        
        if not chunks: return None, None
        
        embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index, chunks
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None, None

# Initialize default store
default_index, default_chunks = create_vector_store(DEFAULT_PDF_PATH)

def generate_questions_from_text(chunks):
    """Generate 5 suggested questions based on the document content."""
    if not chunks: return []
    
    # Take a sample of text (first 2000 chars)
    context = " ".join(chunks[:3])[:2000]
    
    prompt = f"""Based on the following text, generate 5 short, relevant questions that a user might ask. 
    Format them as a simple list. Do not number them. Just 5 questions, one per line.
    
    Text: {context}"""
    
    try:
        response = call_ollama_non_stream([{"role": "user", "content": prompt}])
        # Process response into list
        questions = [q.strip("- ").strip() for q in response.split("\n") if q.strip() and "?" in q]
        return questions[:5]
    except:
        return ["What is this document about?", "Summarize the key points."]
        


def retrieve_context(query, index, chunks):
    if index is None or not chunks or not embedder: return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, 5)
    
    docs = []
    for i, score in zip(I[0], D[0]):
        # Check bound
        if i < len(chunks) and score >= SIMILARITY_THRESHOLD:
            docs.append(chunks[i])
    return docs

# ==========================================
# SYSTEM PROMPTS
# ==========================================
TASK1_SYSTEM_PROMPT = """Act as a professional tutor and exam-oriented instructor.

Answer questions based on their type:
- For short-type questions: give a precise, to-the-point answer in 1–2 lines.
- For long or brief questions: explain clearly in 5–6 lines with proper structure.

Guidelines:
- Use simple, professional language.
- Be clear, factual, and easy to revise.
- Avoid unnecessary filler or storytelling.
- Focus on definitions, key points, and clarity.
- Make answers suitable for exams and academic understanding.

Maintain a calm, teacher-like tone in every response."""

TASK2_SYSTEM_PROMPT = """Act as a professional tutor and exam-oriented instructor.

The model must always identify itself as:
"LUCA from 10X Technologies"

Answer questions based on their type:
- For short-type questions: give a precise, to-the-point answer in 1–2 lines.
- For long or brief questions: explain clearly in 5–6 lines with proper structure.

Guidelines:
- Use simple, professional language.
- Be clear, factual, and easy to revise.
- Avoid unnecessary filler or storytelling.
- Focus on definitions, key points, and clarity.
- Make answers suitable for exams and academic understanding.

Tone:
- Calm, teacher-like, and professional.
- Responses should reflect expertise and clarity."""

# ==========================================
# TASK HANDLERS
# ==========================================

# --- Task 1: Open Source Models ---
def task1_text_response(message, history):
    messages = normalize_history(history)
    # Add custom system prompt for Task 1
    messages.insert(0, {"role": "system", "content": TASK1_SYSTEM_PROMPT})
    
    # Ensure message is a clean string
    messages.append({"role": "user", "content": _force_string(message)})
    
    try:
        stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
        partial_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            partial_response += content
            yield partial_response
        return
    except Exception as e:
        yield f"Error: {e}"

# --- Task 1B: Speech ---
def task1_speech_response(audio_path):
    if not audio_path: return None, "No audio detected."
    
    try:
        # STT
        transcribed = whisper_model.transcribe(audio_path)["text"].strip()
    except: return None, "Error in transcription."

    # Process via helper 
    # Use Task 1 System Prompt
    messages = [{"role": "system", "content": TASK1_SYSTEM_PROMPT}, {"role": "user", "content": transcribed}]
    llm_response = call_ollama_non_stream(messages)

    # TTS
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

    return audio_out, f"**User:** {transcribed}\n\n**AI:** {llm_response}"


# --- Task 2: LUCA Assistant (Identity Enforced) ---
def is_identity_question(text):
    """Check if the question is asking about identity."""
    import string
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    
    identity_patterns = [
        "who are you",
        "who r you", 
        "what are you",
        "who made you",
        "who created you",
        "who developed you",
        "who built you",
        "what is your name",
        "whats your name",
        "tell me about yourself"
    ]
    
    is_identity = any(pattern in text_clean for pattern in identity_patterns)
    print(f"DEBUG Identity Check: '{text}' -> cleaned: '{text_clean}' -> is_identity: {is_identity}")
    return is_identity

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
                print(f"DEBUG: Transcribed audio: '{user_input}'")
        except Exception as e:
            return f"Transcription error: {str(e)}", None, None
    
    user_input = _force_string(user_input)
    if not user_input:
        return "Please say or type something.", None, None

    print(f"DEBUG: Processing input: '{user_input}'")
    
    # 2. Logic & Identity Check (Hardcoded override) - THIS MUST HAPPEN FIRST
    if is_identity_question(user_input):
        print("DEBUG: Identity question detected! Returning LUCA identity.")
        response_text = LUCA_IDENTITY_TEXT
    else:
        print("DEBUG: Regular question, calling LLM with LUCA Prompt...")
        # Standard chat with System Prompt
        msgs = normalize_history(history)
        msgs.insert(0, {"role": "system", "content": TASK2_SYSTEM_PROMPT})
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
        
    return response_text, audio_out


# --- Task 3: School RAG ---
def process_upload(file_path):
    if not file_path:
        # Revert to default
        idx, chk = default_index, default_chunks
        questions = generate_questions_from_text(chk)
        # Return state and default samples
        samples = [[q] for q in questions]
        return (idx, chk), gr.Dataset(samples=samples), ""
    
    # Check size
    if os.path.getsize(file_path) > 2 * 1024 * 1024:
        # Too big
        return None, gr.Dataset(), "Error: File size exceeds 2MB limit."

    idx, chk = create_vector_store(file_path)
    if not idx:
        return None, gr.Dataset(), "Error processing PDF."
    
    # Generate questions
    questions = generate_questions_from_text(chk)
    samples = [[q] for q in questions]
    
    return (idx, chk), gr.Dataset(samples=samples), "PDF Processed Successfully!"

def task3_rag_response(message, history, rag_state):
    # Determine context: Uploaded or Default
    if rag_state:
        index, chunks = rag_state
    else:
        index, chunks = default_index, default_chunks

    if not index:
        yield "PDF not loaded or processed correctly."
        return

    message = _force_string(message)
    docs = retrieve_context(message, index, chunks)
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
footer {visibility: hidden !important; display: none !important;}
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
                    gr.Audio(label="LUCA Voice Reply", autoplay=True)
                ],
                title="Talk to LUCA",
                description="Ask 'Who are you?' to verify identity."
            )

        # TASK 3
        with gr.Tab("Task 3: School AI (RAG)"):
            gr.Markdown("### RAG-Based School Assistant")
            gr.Markdown("Objective: Answer domain-specific questions from the School PDF. Rejects unrelated queries.")
            
            # State for storing vector store (index, chunks)
            rag_state = gr.State()
            
            with gr.Row():
                t3_file = gr.File(label="Upload Custom PDF (Max 2MB)", file_types=[".pdf"], file_count="single")
                t3_status = gr.Markdown("")

            t3_chat = gr.Chatbot(height=400, label="School Bot")
            
            # Dynamic suggestions using Dataset
            # Initial samples from default text
            init_qs = generate_questions_from_text(default_chunks)
            t3_examples = gr.Dataset(components=[t3_msg], label="Suggested Questions", samples=[[q] for q in init_qs])
            
            t3_msg = gr.Textbox(label="Question", placeholder="Ask about the document...")
            
            with gr.Row():
                t3_sub = gr.Button("Ask")
                t3_clr = gr.Button("Clear Chat")
            
            # Handle File Upload
            t3_file.change(
                process_upload, 
                inputs=[t3_file], 
                outputs=[rag_state, t3_examples, t3_status]
            )
            
            # Handle Click on Example
            def on_example_click(x):
                return x[0]
            
            t3_examples.click(on_example_click, inputs=[t3_examples], outputs=[t3_msg])

            # Handler for chatbot
            def respond_t3(msg, hist, state):
                if not msg.strip(): 
                    return hist
                
                if hist is None: hist = []
                
                full_response = ""
                for partial in task3_rag_response(msg, hist, state):
                    full_response = partial
                
                new_hist = hist + [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": full_response}
                ]
                return new_hist
            
            t3_msg.submit(respond_t3, [t3_msg, t3_chat, rag_state], [t3_chat])
            t3_sub.click(respond_t3, [t3_msg, t3_chat, rag_state], [t3_chat])
            t3_clr.click(lambda: [], None, [t3_chat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
