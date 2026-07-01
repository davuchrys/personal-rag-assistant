import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables early so the pipeline can access GEMINI_API_KEY
load_dotenv()

from src.rag_pipeline import RAGPipeline

# --- Setup ---
st.set_page_config(page_title="Personal RAG Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# --- Custom CSS for Modern, Interactive UI ---
st.markdown("""
<style>
    /* Global styles and typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Soft gradient background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Header styling with gradient text */
    .hero-container {
        padding: 3rem 0 2rem 0;
        text-align: center;
        animation: fadeInUp 0.8s ease-out;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #c026d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    
    /* Cards styling with glassmorphism and hover effects */
    .feature-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(255,255,255,0.8);
        text-align: center;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out;
        animation-fill-mode: both;
    }
    
    .feature-card:nth-child(1) { animation-delay: 0.1s; }
    .feature-card:nth-child(2) { animation-delay: 0.2s; }
    .feature-card:nth-child(3) { animation-delay: 0.3s; }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: #3b82f6;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        transition: transform 0.3s ease;
    }
    .feature-card:hover .feature-icon {
        transform: scale(1.1) rotate(5deg);
    }
    
    .feature-title {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .feature-text {
        font-size: 0.95rem;
        color: #64748b;
    }
    
    /* Answer labels */
    .answer-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Refusal warning box */
    .refusal-box {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        color: #991b1b;
        margin-bottom: 1rem;
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Interactive Sources badges */
    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
        color: #334155;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #cbd5e1;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .source-badge:hover {
        background: linear-gradient(135deg, #e0e7ff, #c7d2fe);
        color: #3730a3;
        border-color: #818cf8;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }
    
    /* Customizing Streamlit Button */
    .stButton > button {
        transition: all 0.3s ease !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4) !important;
        background: linear-gradient(135deg, #1d4ed8, #4338ca) !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Fade in main vertical blocks */
    div[data-testid="stVerticalBlock"] {
        animation: fadeInUp 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📂 Document Workspace")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=['pdf', 'txt', 'md'], 
        accept_multiple_files=True,
        help="Upload PDF, TXT, or Markdown files here."
    )
    
    if uploaded_files:
        st.markdown("**Uploaded Files:**")
        for f in uploaded_files:
            st.markdown(f"- `{f.name}`")
            
        st.caption("Documents must be indexed before asking questions.")
        
        if st.button("Index Documents", type="primary", use_container_width=True):
            file_paths = []
            for file in uploaded_files:
                file_path = os.path.join("data", file.name)
                with open(file_path, "wb") as f_out:
                    f_out.write(file.getbuffer())
                file_paths.append(file_path)
            
            with st.spinner("Indexing documents..."):
                num_chunks = pipeline.ingest_files(file_paths)
                st.success(f"Indexed {len(uploaded_files)} files ({num_chunks} chunks).")
                st.session_state['indexed'] = True
                
    st.divider()
    st.markdown("### ⚙️ Settings")
    distance_threshold = st.slider(
        "Distance Threshold", 
        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="Lower value means stricter matching (fewer chunks). Higher value allows weaker matches."
    )
    debug_mode = st.checkbox("Debug Mode", help="Show retrieved chunks and distance scores.")

# --- Main Content ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Personal RAG Assistant</div>
    <div class="hero-subtitle">Ask questions from your own documents with source-grounded answers.</div>
    <div class="hero-desc">This assistant only answers questions based on the exact information found in the documents you upload. If it doesn't know, it will tell you.</div>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card" style="animation-delay: 0.1s;">
        <div class="feature-icon">📄</div>
        <div class="feature-title">Upload Documents</div>
        <div class="feature-text">Securely process your PDFs, TXT, and MD files locally.</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card" style="animation-delay: 0.2s;">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Retrieve Evidence</div>
        <div class="feature-text">Advanced semantic search to find relevant context.</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card" style="animation-delay: 0.3s;">
        <div class="feature-icon">💡</div>
        <div class="feature-title">Answer with Sources</div>
        <div class="feature-text">Get AI-generated answers clearly grounded in your text.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Question Area
with st.container(border=True):
    st.markdown("### ✨ What would you like to know?")
    
    db_count = pipeline.get_document_count()
    if not uploaded_files and not st.session_state.get('indexed', False) and db_count == 0:
        st.info("👋 Upload and index a document in the sidebar to begin.")
    elif db_count > 0 and not st.session_state.get('indexed', False):
        st.success(f"📚 Database loaded with {db_count} existing chunks! You can start asking questions.")
        
    query = st.text_area("Question", placeholder="Ask something from your uploaded documents...", label_visibility="collapsed")
    submit_button = st.button("Ask Assistant", type="primary")

if submit_button:
    if not query:
        st.warning("Please enter a question to ask.")
    else:
        with st.spinner("Analyzing documents..."):
            result = pipeline.ask(query=query, distance_threshold=distance_threshold)
            answer = result["answer"]
            chunks = result["context_chunks"]
            
            # Answer Area
            with st.container(border=True):
                fallback_message = "I could not find enough information in the uploaded documents."
                if fallback_message in answer:
                    st.markdown(f'''
                    <div class="refusal-box">
                        <strong>Notice:</strong> {fallback_message}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="answer-label">📝 Answer</div>', unsafe_allow_html=True)
                    st.markdown(answer)
                    
                    if chunks:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="answer-label">📚 Sources</div>', unsafe_allow_html=True)
                        
                        # Extract unique source filenames
                        sources = set(chunk['metadata'].get('filename', 'Unknown') for chunk in chunks)
                        sources_html = "".join([f'<span class="source-badge">📄 {src}</span>' for src in sources])
                        st.markdown(sources_html, unsafe_allow_html=True)
            
            # Debug Area
            if debug_mode:
                st.markdown("### 🛠️ Debug: Retrieved Chunks")
                if not chunks:
                    st.info("No chunks were retrieved (vector DB might be empty or all chunks exceeded the distance threshold).")
                else:
                    for i, chunk in enumerate(chunks):
                        filename = chunk['metadata'].get('filename', 'Unknown')
                        dist = chunk['distance']
                        with st.expander(f"Chunk {i+1} | Source: {filename} | Distance: {dist:.4f}"):
                            st.markdown(f"**Text:**\n\n{chunk['text']}")
