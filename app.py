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
import sys

# Add Translatotron to path
sys.path.append(os.path.join(os.path.dirname(__file__), "translatotron_src"))
try:
    from translatotron.translatotron import Translatotron
    from torchaudio import transforms as T
    import torchaudio
    TRANSLATOTRON_AVAILABLE = True
except Exception as e:
    print(f"Translatotron import failed: {e}")
    TRANSLATOTRON_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
LLM_MODEL = "gemma3:1b"
WHISPER_MODEL_SIZE = "tiny.en"
TTS_VOICE = "en-US-AriaNeural"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PDF_PATH = "school_information.pdf"

# Task 2 Configuration (Restored)
LUCA_IDENTITY_TEXT = "I am LUCA from 10X Technologies."

# Task 3 Configuration
CHUNK_SIZE = 1000  # Increased for better context capture
CHUNK_OVERLAP = 200
SIMILARITY_THRESHOLD = 0.35  # Broader retrieval, relying on strict LLM filtering
REJECTION_RESPONSE = "I am not aware."

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

# 3. Translatotron (Task 1B)
print("Loading Translatotron...")
translatotron_model = None
mel_converter = None
if TRANSLATOTRON_AVAILABLE:
    try:
        translatotron_model = Translatotron()
        translatotron_model.eval()
        # Mel spectrogram converter (matches train.py config)
        mel_converter = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80)
    except Exception as e:
        print(f"Error initializing Translatotron: {e}")

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

# Helper to clean text for TTS (remove markdown and emojis)
def clean_text_for_tts(text):
    """Remove markdown formatting and emojis for better TTS output"""
    import re
    
    # Remove emojis (Unicode emoji ranges)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove markdown bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown lists
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text

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
    
    # 1. Vector Search
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    k_vector = 10
    D, I = index.search(q_emb, k_vector)
    
    retrieved_indices = set()
    debug_info = []

    # Collect Vector Results
    for i, score in zip(I[0], D[0]):
        if i < len(chunks) and score >= SIMILARITY_THRESHOLD:
            retrieved_indices.add(i)
            debug_info.append(f"Vector Match: Chunk {i} (Score: {score:.4f})")

    # 2. Keyword Fallback (Simple string matching)
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    
    if query_words:
        for idx, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            matches = sum(1 for w in query_words if w in chunk_lower)
            if matches >= 1:
                if idx not in retrieved_indices:
                    retrieved_indices.add(idx)
                    debug_info.append(f"Keyword Match: Chunk {idx} (Matches: {matches})")
    
    # Debug Print
    print(f"DEBUG: Query '{query}'")
    for info in debug_info:
        print(f"  - {info}")
        
    # Convert to sorted list
    final_indices = sorted(list(retrieved_indices))
    
    return [chunks[i] for i in final_indices]

def perform_calculation_on_text(query, context_text):
    """
    Analyzes the query and context to perform calculations if requested.
    Returns a system note string with the result, or empty string.
    """
    query_lower = query.lower()
    operation = None
    
    if "average" in query_lower or "mean" in query_lower:
        operation = "average"
    elif "sum" in query_lower or "total" in query_lower:
        operation = "sum"
    elif "max" in query_lower or "highest" in query_lower:
        operation = "max"
    elif "min" in query_lower or "lowest" in query_lower:
        operation = "min"
    
    if not operation:
        return ""

    print(f"DEBUG: Calculation requested ({operation}) for query: '{query}'")

    # specific extraction prompt
    extraction_prompt = f"""
    Analyze the text below and extract all numerical values that match the user's query regarding "{query}".
    - Ignore dates, years (like 2023), or phone numbers.
    - EXTRACT ONLY the specific values (e.g., marks, fees, scores).
    - Return ONLY the numbers separated by commas.
    - If no relevant numbers are found, return NOTHING.
    
    Context:
    {context_text}
    """
    
    try:
        # Re-use the existing helper
        extracted_text = call_ollama_non_stream([{"role": "user", "content": extraction_prompt}])
        
        # Parse numbers using regex
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", extracted_text)
        # Filter out likely years/dates if they look like integers > 1900 and < 2100? 
        # Actually proper prompting is better. Let's rely on the LLM + regex.
        
        numbers = []
        for m in matches:
             try:
                 val = float(m)
                 # Simple heuristic to avoid years if the query is about marks (0-100 usually)
                 # But fees could be large. Let's trust the LLM's extraction for now.
                 numbers.append(val)
             except:
                 pass
        
        if not numbers:
            print("DEBUG: No numbers found for calculation.")
            return ""

        result = 0
        op_name = ""
        
        if operation == "average":
            result = sum(numbers) / len(numbers)
            op_name = "Average"
        elif operation == "sum":
            result = sum(numbers)
            op_name = "Sum"
        elif operation == "max":
            result = max(numbers)
            op_name = "Maximum"
        elif operation == "min":
            result = min(numbers)
            op_name = "Minimum"
            
        print(f"DEBUG: Calculated {op_name}: {result} from {numbers}")
        
        return (f"\n[SYSTEM: The user asked for a calculation. "
                f"I have extracted the following values from the text: {numbers}. "
                f"The calculated {op_name} is {result:.2f}. "
                f"State this result clearly in your answer.]\n")

    except Exception as e:
        print(f"Error during calculation: {e}")
        return ""

# ==========================================
# SYSTEM PROMPTS
# ==========================================
# System prompts removed as per requirement.


# ==========================================
# TASK HANDLERS
# ==========================================

# --- Task 1: Open Source Models ---
def task1_text_response(message, history):
    messages = normalize_history(history)
    # System prompt removed
    
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
    # System prompt removed
    messages = [{"role": "user", "content": transcribed}]
    llm_response = call_ollama_non_stream(messages)

    # Clean response for TTS (remove markdown and emojis)
    tts_text = clean_text_for_tts(llm_response)

    # TTS
    async def get_tts(text):
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            await edge_tts.Communicate(text, TTS_VOICE).save(path)
            return path
        except: return None

    try:
        audio_out = asyncio.run(get_tts(tts_text))
    except RuntimeError:
        audio_out = asyncio.get_event_loop().run_until_complete(get_tts(tts_text))

    return audio_out, f"**User:** {transcribed}\n\n**AI:** {llm_response}"


# --- Task 1B [NEW]: Direct Speech-to-Speech (Translatotron) ---
def task1_translatotron_response(audio_path):
    if not TRANSLATOTRON_AVAILABLE or translatotron_model is None:
        return None, "Translatotron model not available."
        
    if not audio_path:
        return None, "No audio input."

    try:
        # 1. Load Audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 2. Resample to 16k if needed
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            
        # 3. Mixing to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # 4. Generate Mel Spectrogram
        # Input to mel_converter: (channel, time) -> Output: (channel, n_mels, time)
        mel = mel_converter(waveform)
        
        # 5. Prepare for Model
        # Model expects (batch, time, n_mels)
        # Current mel is (1, 80, time) -> Permute to (1, time, 80)
        input_mel = mel.permute(0, 2, 1)
        
        # 6. Inference
        with torch.no_grad():
            output_waveform, _, _, _ = translatotron_model(input_mel)
            # output_waveform is (batch, channels, time) -> (1, 1, time) usually from GriffinLim
            
        # 7. Save output
        out_wav = output_waveform.squeeze().cpu().numpy()
        
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        # Helper to save (using scipy or soundfile, checking what's available)
        import soundfile as sf
        sf.write(path, out_wav, 16000)
        
        return path, "Processing complete via Translatotron (Direct S2S)."
        
    except Exception as e:
        print(f"Translatotron Error: {e}")
        return None, f"Error processing audio: {e}"


# --- Task 2: LUCA Assistant (Identity Enforced) ---
def is_identity_question(text):
    """Check if the question is asking about identity using regex."""
    text_clean = text.lower().strip()
    
    # Regex patterns for more robust matching
    patterns = [
        r"who\s+(are|r)\s+you",
        r"what\s+are\s+you",
        r"who\s+(made|created|built|developed|designed)\s+you",
        r"(what.?s|what\s+is)\s+your\s+name",
        r"(tell|say)\s+.*your\s+name",
        r"call\s+you",
        r"tell\s+me\s+about\s+yourself",
        r"introduce\s+yourself",
        r"your\s+identity"
    ]
    
    for pattern in patterns:
        if re.search(pattern, text_clean):
            print(f"DEBUG Identity Check: MATCHED pattern '{pattern}' in '{text_clean}'")
            return True
            
    print(f"DEBUG Identity Check: NO MATCH for '{text_clean}'")
    return False

# --- Task 2: LUCA Assistant (Text Chat) ---
def task2_text_chat(message, history):
    """Text-based chat with LUCA (streaming)"""
    user_input = _force_string(message)
    if not user_input:
        yield "Please type something."
        return
    
    print(f"DEBUG: Processing text input: '{user_input}'")
    
    # Identity Check (Hardcoded override)
    if is_identity_question(user_input):
        print("DEBUG: Identity question detected! Returning LUCA identity.")
        yield LUCA_IDENTITY_TEXT
        return
    
    # Standard chat (No System Prompt)
    msgs = normalize_history(history)
    
    msgs.append({"role": "user", "content": user_input})
    
    try:
        stream = ollama.chat(model=LLM_MODEL, messages=msgs, stream=True)
        partial_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            partial_response += content
            yield partial_response
    except Exception as e:
        yield f"Error: {e}"

# --- Task 2: LUCA Assistant (Voice) ---
def task2_voice_handler(audio_path):
    """Voice-based interaction with LUCA"""
    if not audio_path:
        return None, "No audio detected."
    
    try:
        # STT
        transcribed = whisper_model.transcribe(audio_path)["text"].strip()
        print(f"DEBUG: Transcribed audio: '{transcribed}'")
    except Exception as e:
        return None, f"Transcription error: {str(e)}"
    
    # Identity Check (Hardcoded override)
    if is_identity_question(transcribed):
        print("DEBUG: Identity question detected! Returning LUCA identity.")
        response_text = LUCA_IDENTITY_TEXT
    else:
        print("DEBUG: Regular question, calling LLM with LUCA Prompt...")
        messages = [
            {"role": "user", "content": transcribed}
        ]
        response_text = call_ollama_non_stream(messages)
    
    # Clean response for TTS (remove markdown and emojis)
    tts_text = clean_text_for_tts(response_text)
    
    # TTS
    async def get_tts(text):
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            await edge_tts.Communicate(text, TTS_VOICE).save(path)
            return path
        except:
            return None
    
    try:
        audio_out = asyncio.run(get_tts(tts_text))
    except RuntimeError:
        audio_out = asyncio.get_event_loop().run_until_complete(get_tts(tts_text))
    
    return audio_out, f"**User:** {transcribed}\n\n**LUCA:** {response_text}"


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

    context_str = "\n\n".join([f"Excerpt {idx+1}:\n{d}" for idx, d in enumerate(docs)])
    print(f"DEBUG: LLM Context ({len(docs)} chunks)\n")
    
    # Check for calculation
    calc_note = perform_calculation_on_text(message, context_str)

    prompt = (
        "You are a strictly factual assistant. Use ONLY the provided context excerpts to answer the question.\n\n"
        "CONTEXT:\n"
        f"{context_str}\n\n"
        f"QUESTION: {message}\n"
        f"{calc_note}\n\n"
        "STRICT INSTRUCTIONS:\n"
        "1. **Analyze & Reason**: The answer may not be in a single sentence. Combine information from different parts of the context context to form a complete answer.\n"
        "2. **Logical Inference**: You can infer answers (e.g., calculating totals, comparing values) ONLY if the data supports it.\n"
        "3. **Evidence**: Support your answer with specific facts or numbers from the text.\n"
        "4. **Strict Boundary**: If the information is missing or cannot be logically inferred from the context, YOU MUST REPLY: 'I am not aware.'\n"
        "5. **No Outside Knowledge**: Do not use external facts (e.g., general world knowledge) not present in the excerpts."
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
            
            with gr.Accordion("Sub-Task B: Direct Speech-to-Speech (Translatotron)", open=False):
                gr.Markdown("""
                **Model**: Translatotron (End-to-End Direct Speech-to-Speech)
                **Status**: Research Architecture Demo (Random Weights)
                **Note**: Since this is a raw model architecture without pre-trained weights for a specific language pair, the output will be processed audio noise/gibberish. This demonstrates the *pipeline execution* of a direct S2S model as requested.
                """)
                with gr.Row():
                    t1b_audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
                    t1b_audio_out = gr.Audio(label="Direct S2S Output", autoplay=True)
                t1b_status = gr.Markdown(label="Status")
                t1b_btn = gr.Button("Run Translatotron")
                
                t1b_btn.click(task1_translatotron_response, t1b_audio_in, [t1b_audio_out, t1b_status])

            with gr.Accordion("Sub-Task C: Pipeline Speech-to-Speech (Whisper + Gemma + EdgeTTS)", open=False):
                gr.Markdown("**Legacy Approach**: Uses STT -> LLM -> TTS pipeline.")
                with gr.Row():
                    t1_audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
                    t1_audio_out = gr.Audio(label="Output Audio", autoplay=True)
                t1_transcript = gr.Markdown(label="Transcript")
                t1_btn = gr.Button("Run Pipeline")
                
                t1_btn.click(task1_speech_response, t1_audio_in, [t1_audio_out, t1_transcript])

        # TASK 2
        with gr.Tab("Task 2: LUCA Voice Assistant"):
            gr.Markdown("### LUCA Voice Assistant")
            gr.Markdown("Objective: Voice-enabled assistant identifying as 'LUCA from 10X Technologies'.")
            
            with gr.Accordion("Text Chat with LUCA", open=True):
                gr.ChatInterface(
                    task2_text_chat,
                    title="Chat with LUCA",
                    description="Ask 'Who are you?' to verify identity."
                )
            
            with gr.Accordion("Voice Interaction with LUCA", open=False):
                with gr.Row():
                    t2_audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak to LUCA")
                    t2_audio_out = gr.Audio(label="LUCA's Voice Response", autoplay=True)
                t2_transcript = gr.Markdown(label="Conversation")
                t2_voice_btn = gr.Button("Send Voice Message")
                
                t2_voice_btn.click(task2_voice_handler, t2_audio_in, [t2_audio_out, t2_transcript])

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
            # Define Textbox first (for reference) but do not render yet
            t3_msg = gr.Textbox(label="Question", placeholder="Ask about the document...", render=False)
            
            # Initial samples from default text
            init_qs = generate_questions_from_text(default_chunks)
            t3_examples = gr.Dataset(components=[t3_msg], label="Suggested Questions", samples=[[q] for q in init_qs])
            
            # Render Textbox now
            t3_msg.render()
            
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
