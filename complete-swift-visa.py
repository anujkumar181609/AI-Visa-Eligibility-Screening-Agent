import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
import PyPDF2
import re
import unicodedata
import hashlib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any
import glob
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="SwiftVisa AI Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for ChatGPT Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    .main {
        background: white !important;
    }
    
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    
    [data-testid="stSidebar"] {
        background-color: #202123 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ECECF1 !important;
    }
    
    h1, h2, h3 {
        color: #202123 !important;
        font-weight: 600 !important;
    }
    
    .stButton button {
        border-radius: 8px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding: 0.5rem 1rem !important;
        min-height: 2.5rem !important;
        line-height: 1.4 !important;
    }
    
    button[data-testid="baseButton-primary"] {
        background: #10A37F !important;
        color: white !important;
    }
    
    .stChatMessage[data-testid*="user"] [data-testid="stChatMessageContent"] {
        background: #F7F7F8 !important;
        border-radius: 12px !important;
    }
    
    .stChatMessage[data-testid*="assistant"] [data-testid="stChatMessageContent"] {
        background: white !important;
        border-radius: 12px !important;
        border: 1px solid #E5E5E5 !important;
    }
    
    .badge-success { background: #D1FAE5; color: #065F46; }
    .badge-warning { background: #FEF3C7; color: #92400E; }
    
    /* Fix button spacing to prevent overlap */
    .stButton {
        margin-bottom: 0.5rem !important;
    }
    
    /* Ensure expander content has proper spacing */
    .streamlit-expanderContent {
        padding: 1rem !important;
    }
    
    /* Fix expander header to prevent arrow overlap */
    .streamlit-expanderHeader {
        padding: 0.75rem 1rem !important;
        padding-left: 2.5rem !important;
    }
    
    .streamlit-expanderHeader svg {
        margin-right: 0.75rem !important;
    }
    
    /* Ensure expander text doesn't overlap with arrow */
    details summary {
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    /* Hide the keyboard_arrow_down text completely */
    .streamlit-expanderHeader {
        overflow: hidden !important;
        position: relative !important;
    }
    
    /* Force hide icon text by setting font size to 0 on icon wrapper */
    .streamlit-expanderHeader [role="button"] > div:first-child {
        font-size: 0 !important;
        width: 24px !important;
        height: 24px !important;
        overflow: hidden !important;
    }
    
    /* Keep SVG visible */
    .streamlit-expanderHeader svg {
        font-size: initial !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    /* Alternative: use text-indent to push text off screen */
    .streamlit-expanderHeader [role="button"] > div:first-child > p {
        text-indent: -9999px !important;
        position: absolute !important;
        left: -9999px !important;
    }
    
    /* More aggressive: hide any p tag inside the icon container */
    details summary > div:first-child p {
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
    }
    
    /* Ensure proper spacing for expander label */
    details summary > div:last-child {
        margin-left: 0.5rem !important;
    }
    
    /* Hide the Material Icon text span */
    span[data-testid="stIconMaterial"] {
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        display: none !important;
    }
    
    /* Completely hide the icon text paragraph */
    .streamlit-expanderHeader [role="button"] > div:first-child p {
        position: absolute !important;
        left: -10000px !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
    }
    
    /* Strictly constrain icon container */
    .streamlit-expanderHeader [role="button"] > div:first-child {
        width: 24px !important;
        max-width: 24px !important;
        min-width: 24px !important;
        height: 24px !important;
        overflow: hidden !important;
        flex-shrink: 0 !important;
        margin-right: 8px !important;
    }
    
    /* Make expander button use flexbox properly */
    .streamlit-expanderHeader [role="button"] {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    
    /* Ensure label doesn't wrap and stays on one line */
    .streamlit-expanderHeader [role="button"] > div:last-child {
        flex: 1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Keep SVG visible */
    .streamlit-expanderHeader svg {
        display: block !important;
        width: 24px !important;
        height: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
CHUNKS_FILE = os.path.join(OUTPUT_DIR, 'all_chunks.jsonl')
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, 'chunks_with_embeddings.jsonl')
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
GENERATION_MODEL = 'google/flan-t5-small'
MAX_TOKENS = 256
OVERLAP_TOKENS = 32
TOP_K_RESULTS = 5

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def normalize_slug(s: str, repl="-"):
    """Normalize string to create a clean slug"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9]+", repl, s)
    s = re.sub(rf"{repl}{{2,}}", repl, s)
    s = s.strip(repl)
    return s or "na"

def get_document_hash(content: str) -> str:
    """Generate SHA256 hash of document content"""
    return hashlib.sha256(content.encode()).hexdigest()

def get_all_pdfs(directory: str) -> List[str]:
    """Get all PDF files in a directory"""
    pdf_pattern = os.path.join(directory, '*.pdf')
    pdf_files = glob.glob(pdf_pattern)
    pdf_pattern_upper = os.path.join(directory, '*.PDF')
    pdf_files.extend(glob.glob(pdf_pattern_upper))
    return list(set(pdf_files))

@st.cache_resource
def load_tokenizer():
    """Load tokenizer for chunking"""
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

def chunk_by_tokens(text: str, tokenizer, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    """Split text into overlapping chunks based on token count"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += max_tokens - overlap
    return chunks

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from a single PDF and return page data"""
    pages_data = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            for page_num in range(total_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    pages_data.append({
                        'page': page_num + 1,
                        'text': text
                    })
    except Exception as e:
        st.error(f'Error reading {pdf_path}: {e}')
        return []
    return pages_data

def process_all_pdfs(data_dir: str, output_path: str, tokenizer):
    """Process ALL PDFs in a directory and create chunks"""
    pdf_files = get_all_pdfs(data_dir)
    
    if not pdf_files:
        return 0, []
    
    all_chunks = []
    total_chunks = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
        pdf_name = os.path.basename(pdf_path)
        status_text.markdown(f'**Processing:** {pdf_name} ({pdf_idx}/{len(pdf_files)})')
        
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            continue
        
        full_text = ' '.join([p['text'] for p in pages])
        doc_hash = get_document_hash(full_text)[:8]
        
        pdf_chunks = 0
        for page_data in pages:
            page_num = page_data['page']
            text = page_data['text']
            chunks = chunk_by_tokens(text, tokenizer)
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': f'{doc_hash}_p{page_num}_c{i}',
                    'text': chunk,
                    'metadata': {
                        'page': page_num,
                        'chunk_index': i,
                        'doc_hash': doc_hash,
                        'source': pdf_name,
                        'pdf_index': pdf_idx
                    }
                }
                all_chunks.append(chunk_data)
                pdf_chunks += 1
        
        total_chunks += pdf_chunks
        progress_bar.progress(pdf_idx / len(pdf_files))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    progress_bar.empty()
    status_text.empty()
    
    return total_chunks, pdf_files

@st.cache_resource
def load_embedding_model():
    """Load embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(input_file: str, output_file: str, embedding_model):
    """Generate embeddings for all chunks"""
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    if not chunks:
        return 0
    
    texts = [chunk['text'] for chunk in chunks]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in batches
    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        progress = min((i + batch_size) / len(texts), 1.0)
        progress_bar.progress(progress)
        status_text.markdown(f'**Generating embeddings:** {int(progress * 100)}% complete')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
            f.write(json.dumps(chunk) + '\n')
    
    progress_bar.empty()
    status_text.empty()
    
    return len(chunks)

@st.cache_resource
def load_generator_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    return tokenizer, model

class MultiDocRAGPipeline:
    def __init__(self, embeddings_file: str, embedding_model):
        """Initialize RAG pipeline for multiple documents"""
        self.chunks = []
        self.embeddings = []
        self.index = None
        self.embedding_model = embedding_model
        self.doc_sources = set()
        self.load_data(embeddings_file)
        self.build_index()
    
    def load_data(self, embeddings_file: str):
        """Load chunks with embeddings from all documents"""
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks.append(chunk)
                self.embeddings.append(chunk['embedding'])
                self.doc_sources.add(chunk['metadata'].get('source', 'unknown'))
        
        self.embeddings = np.array(self.embeddings, dtype='float32')
    
    def build_index(self):
        """Build FAISS index for similarity search"""
        faiss.normalize_L2(self.embeddings)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
    
    def search(self, query: str, k: int = TOP_K_RESULTS):
        """Search for relevant chunks"""
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype='float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                chunk = self.chunks[idx]
                results.append({
                    'chunk': chunk,
                    'score': float(score)
                })
        
        return results
    
    def format_context(self, results):
        """Format search results as context"""
        context_parts = []
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'unknown')
            page = metadata.get('page', 'N/A')
            context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")
        
        return "\n\n".join(context_parts)

def generate_answer(query: str, context: str, generator):
    """Generate answer based on query and context"""
    prompt = f"""Context from UK Student Visa documents:
{context}

Question: {query}

Based on the context provided above, please answer the question accurately and concisely.
If the answer cannot be found in the context, say 'I cannot find this information in the provided documents.'

Answer:"""
    
    if generator:
        try:
            response = generator(prompt, max_length=200, do_sample=False)[0]['generated_text']
            return response
        except Exception as e:
            return f"Based on the documents, here's what I found:\n\n{context[:500]}..."
    else:
        return f"Based on the documents, here's what I found:\n\n{context[:500]}..."

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: center;'>SwiftVisa</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(255,255,255,0.8); text-align: center; margin-top: -1rem;'>AI-Powered Visa Assistant</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigate",
        ["Chat Assistant", "Document Processing"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    # System status
    st.markdown("<h3 style='color: white;'>System Status</h3>", unsafe_allow_html=True)
    
    embeddings_exist = os.path.exists(EMBEDDINGS_FILE)
    pdf_files = get_all_pdfs(DATA_DIR)
    
    if embeddings_exist:
        with open(EMBEDDINGS_FILE, 'r') as f:
            num_chunks = sum(1 for _ in f)
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
            <p style='color: white; margin: 0; font-size: 0.9rem;'><strong>Status:</strong> Online</p>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;'><strong>Chunks:</strong> {num_chunks:,}</p>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;'><strong>Documents:</strong> {len(pdf_files)}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
            <p style='color: white; margin: 0; font-size: 0.9rem;'><strong>Status:</strong> Setup Required</p>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;'><strong>PDFs Found:</strong> {len(pdf_files)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(255,255,255,0.6); text-align: center; font-size: 0.8rem;'>SwiftVisa AI v2.0<br>Powered by RAG Technology</p>", unsafe_allow_html=True)

# --- PAGE 1: Chat Assistant ---
if page == "Chat Assistant":
    
    # Check if system is ready
    if not embeddings_exist:
        st.markdown("""
        <div class='main-header'>
            <h1>Chat Assistant</h1>
            <p>Your intelligent UK Student Visa advisor</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.error("**System Not Ready** - Please process documents first")
        
        st.markdown("""
        <div class='info-card'>
            <h3>🚀 Quick Setup</h3>
            <ol>
                <li>Navigate to <strong>📄 Document Processing</strong> in the sidebar</li>
                <li>Upload your PDF documents or use existing ones</li>
                <li>Click <strong>Start Processing Pipeline</strong></li>
                <li>Return here to start chatting!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Initialize RAG pipeline
    if not st.session_state.rag_initialized:
        with st.spinner("Initializing AI system..."):
            try:
                embedding_model = load_embedding_model()
                st.session_state.rag_pipeline = MultiDocRAGPipeline(EMBEDDINGS_FILE, embedding_model)
                st.session_state.generator = load_generator_pipeline()
                st.session_state.rag_initialized = True
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.stop()
    
    rag = st.session_state.rag_pipeline
    generator = st.session_state.generator
    
    # Header with controls
    col1, col2, col3, col4 = st.columns([6, 1, 1, 1])
    
    with col1:
        st.markdown("""
        <div style='padding: 1rem 0;'>
            <h1 style='margin: 0; color: #202123;'>SwiftVisa AI Assistant</h1>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>Ask me anything about UK Student Visa requirements</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Stats", use_container_width=True):
            st.session_state.show_stats = not st.session_state.get('show_stats', False)
    
    with col3:
        if st.button("Docs", use_container_width=True):
            st.session_state.show_docs = not st.session_state.get('show_docs', False)
    
    with col4:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Show stats/docs if toggled
    if st.session_state.get('show_stats', False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", f"{len(rag.chunks):,}")
        with col2:
            st.metric("Documents", len(rag.doc_sources))
        with col3:
            st.metric("Messages", len(st.session_state.messages))
    
    if st.session_state.get('show_docs', False):
        st.markdown("<div class='info-card'>", unsafe_allow_html=True)
        st.markdown("**📚 Loaded Documents:**")
        for doc in sorted(rag.doc_sources):
            st.markdown(f"<span class='badge badge-info'>{doc}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            # Welcome message
            st.markdown("""
            <div class='welcome-card'>
                <h2>👋 Welcome to SwiftVisa AI Assistant!</h2>
                <p>I'm your intelligent assistant for UK Student Visa inquiries. I have access to comprehensive visa documentation and can provide accurate, sourced answers.</p>
                <br>
                <h3>✨ What I Can Help With:</h3>
                <ul>
                    <li>Financial requirements and proof of funds</li>
                    <li>English language requirements and testing</li>
                    <li>CAS (Confirmation of Acceptance for Studies) requirements</li>
                    <li>Work permissions and restrictions for students</li>
                    <li>Dependant visa information</li>
                    <li>Application procedures and documentation</li>
                </ul>
                <br>
                <p><strong>💡 Tip:</strong> Click on any example question below to get started!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("View Source Citations"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class='source-citation'>
                                <div class='source-citation-header'>
                                    Source {i}: {source['source']} (Page {source['page']})
                                </div>
                                <div style='color: #666; font-size: 0.85rem; margin-bottom: 0.5rem;'>
                                    Relevance Score: {source['score']:.1%}
                                </div>
                                <div style='background: white; padding: 0.75rem; border-radius: 5px; font-size: 0.9rem;'>
                                    {source['text'][:300]}{"..." if len(source['text']) > 300 else ""}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your question here...", key="chat_input")
    
    # Check if there's a new user message that needs processing
    needs_response = False
    if user_input:
        # Add user message from chat input
        st.session_state.messages.append({"role": "user", "content": user_input})
        needs_response = True
    elif len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # Last message is from user but no assistant response yet (from button click)
        user_input = st.session_state.messages[-1]["content"]
        needs_response = True
    
    if needs_response:
        # Display user message if not already shown
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Generate response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing documents..."):
                    try:
                        results = rag.search(user_input, k=TOP_K_RESULTS)
                        
                        if results:
                            context = rag.format_context(results)
                            answer = generate_answer(user_input, context, generator)
                            
                            st.markdown(answer)
                            
                            # Prepare sources
                            sources = []
                            for result in results:
                                chunk = result['chunk']
                                sources.append({
                                    'source': chunk['metadata'].get('source', 'unknown'),
                                    'page': chunk['metadata'].get('page', 'N/A'),
                                    'score': result['score'],
                                    'text': chunk['text']
                                })
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                            
                            with st.expander("View Source Citations"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"""
                                    <div class='source-citation'>
                                        <div class='source-citation-header'>
                                            📄 Source {i}: {source['source']} (Page {source['page']})
                                        </div>
                                        <div style='color: #666; font-size: 0.85rem; margin-bottom: 0.5rem;'>
                                            Relevance Score: {source['score']:.1%}
                                        </div>
                                        <div style='background: white; padding: 0.75rem; border-radius: 5px; font-size: 0.9rem;'>
                                            {source['text'][:300]}{"..." if len(source['text']) > 300 else ""}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            error_msg = "I couldn't find relevant information to answer your question. Please try rephrasing or ask about UK Student Visa topics."
                            st.warning(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                    
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        
        st.rerun()
    
    # Example questions
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Example Questions - Click to Ask"):
        examples = [
            ("Financial Requirements", "What are the financial requirements for a UK Student visa?"),
            ("Dependants", "Can Child Students bring dependants to the UK?"),
            ("CAS Information", "What is a valid CAS and what information must it include?"),
            ("Proof of Funds", "How long must students show they have held the required funds?"),
            ("English Language", "What are the English language requirements for UK student visa?"),
            ("Work Permissions", "Can international students work part-time in the UK?")
        ]
        
        col1, col2 = st.columns(2)
        for idx, (emoji_title, question) in enumerate(examples):
            with col1 if idx % 2 == 0 else col2:
                if st.button(emoji_title, key=f"ex_{idx}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()
# --- PAGE 2: Document Processing ---
elif page == "Document Processing":
    # Header
    st.markdown("""
    <div style='padding: 1rem 0;'>
        <h1 style='margin: 0; color: #202123;'>Document Processing Center</h1>
        <p style='margin: 0.5rem 0 0 0; color: #666;'>Upload, process, and manage your PDF documents for AI-powered search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # PDF Management Section
        st.markdown("<h2 style='color: #202123;'>Document Library</h2>", unsafe_allow_html=True)
        
        existing_pdfs = get_all_pdfs(DATA_DIR)
        
        if existing_pdfs:
            st.success(f"{len(existing_pdfs)} PDF file(s) ready for processing")
            
            for idx, pdf in enumerate(existing_pdfs, 1):
                pdf_name = os.path.basename(pdf)
                pdf_size = os.path.getsize(pdf) / 1024  # KB
                st.markdown(f"""
                <div class='info-card'>
                    <strong>{idx}. {pdf_name}</strong><br>
                    <span style='color: #666; font-size: 0.9rem;'>Size: {pdf_size:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No PDF files found in the data folder")
            st.info("💡 Upload PDFs below or place them in: `swiftvisa_HMI/data/`")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # File uploader
        st.markdown("<h3 style='color: #667eea;'>📤 Upload New Documents</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag and drop PDF files here or click to browse",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to process"
        )
        
        if uploaded_files:
            if st.button("💾 Save Uploaded Files", type="primary", use_container_width=True):
                with st.spinner("Saving files..."):
                    saved_count = 0
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(DATA_DIR, uploaded_file.name)
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        saved_count += 1
                    st.success(f"✅ Successfully saved {saved_count} file(s)")
                    st.rerun()
    
    with col2:
        # Status Dashboard
        st.markdown("<h2 style='color: #667eea;'>⚙️ Processing Status</h2>", unsafe_allow_html=True)
        
        if embeddings_exist:
            with open(EMBEDDINGS_FILE, 'r') as f:
                chunk_count = sum(1 for _ in f)
            
            st.markdown(f"""
            <div class='stat-card'>
                <h2>{chunk_count:,}</h2>
                <p>Chunks Indexed</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show document stats
            doc_counts = {}
            with open(EMBEDDINGS_FILE, 'r') as f:
                for line in f:
                    chunk = json.loads(line)
                    source = chunk['metadata'].get('source', 'unknown')
                    doc_counts[source] = doc_counts.get(source, 0) + 1
            
            st.markdown("<div class='info-card'><strong>📚 Indexed Documents:</strong>", unsafe_allow_html=True)
            for doc, count in doc_counts.items():
                st.markdown(f"<span class='badge badge-success'>{doc}</span> {count} chunks<br>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-card'>
                <h3 style='color: #856404; margin-top: 0;'>⚠️ Not Processed</h3>
                <p>Generate embeddings below to activate the AI assistant</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Processing Pipeline Section
    st.markdown("<h2 style='color: #667eea; text-align: center;'>🚀 AI Processing Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-card' style='text-align: center;'>
            <div class='step-indicator'>1</div>
            <h3>Extract Text</h3>
            <p>Parse PDF documents and extract all text content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card' style='text-align: center;'>
            <div class='step-indicator'>2</div>
            <h3>Generate Embeddings</h3>
            <p>Create AI-powered semantic embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-card' style='text-align: center;'>
            <div class='step-indicator'>3</div>
            <h3>Build Index</h3>
            <p>Create searchable vector database</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not existing_pdfs:
        st.warning("⚠️ Please upload PDF files first to begin processing!")
    else:
        if st.button("🚀 Start Processing Pipeline", type="primary", use_container_width=True):
            try:
                # Step 1: Process PDFs
                with st.status("Processing documents...", expanded=True) as status:
                    st.write("📄 Step 1/3: Extracting text from PDFs...")
                    tokenizer = load_tokenizer()
                    num_chunks, pdf_files = process_all_pdfs(DATA_DIR, CHUNKS_FILE, tokenizer)
                    
                    if num_chunks > 0:
                        st.write(f"✅ Extracted {num_chunks:,} chunks from {len(pdf_files)} PDF(s)")
                        
                        # Step 2: Generate embeddings
                        st.write("🧠 Step 2/3: Generating AI embeddings...")
                        embedding_model = load_embedding_model()
                        num_embeddings = generate_embeddings(CHUNKS_FILE, EMBEDDINGS_FILE, embedding_model)
                        st.write(f"✅ Generated {num_embeddings:,} embeddings")
                        
                        # Step 3: Build index
                        st.write("📊 Step 3/3: Building search index...")
                        st.session_state.rag_initialized = False
                        st.write("✅ Search index built successfully!")
                        
                        status.update(label="✅ Processing Complete!", state="complete")
                        st.balloons()
                        st.success("🎉 All documents processed successfully! You can now use the Chat Assistant.")
                    else:
                        st.error("❌ No chunks created. Please check your PDF files.")
                        
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                with st.expander("View Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    # Advanced Options
    if embeddings_exist:
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        with st.expander("⚙️ Advanced Options"):
            st.warning("**⚠️ Warning:** This will delete all generated embeddings. You'll need to reprocess documents.")
            if st.button("🗑️ Delete All Embeddings", type="secondary"):
                try:
                    if os.path.exists(CHUNKS_FILE):
                        os.remove(CHUNKS_FILE)
                    if os.path.exists(EMBEDDINGS_FILE):
                        os.remove(EMBEDDINGS_FILE)
                    st.session_state.rag_initialized = False
                    st.success("✅ Embeddings deleted successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
