import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline

def format_confidence(distance: float):
    """Return a confidence label and hex color based on distance.
    Lower distance = higher confidence.
    """
    if distance <= 0.45:
        return "High confidence", "#4caf50"
    if distance <= 0.75:
        return "Medium confidence", "#ff9800"
    if distance <= 1.0:
        return "Low confidence", "#f44336"
    return "Insufficient evidence", "#9e9e9e"


# --- Setup ---
st.set_page_config(page_title="RAG Assistant", page_icon="📎", layout="wide")

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()
os.makedirs("data", exist_ok=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Clean, Minimal CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #fafafa;
    }
    
    /* Clean header */
    .app-header {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111;
        margin: 0 0 0.25rem 0;
    }
    .app-header p {
        font-size: 0.95rem;
        color: #666;
        margin: 0;
    }
    
    /* Status bar */
    .status-bar {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #15803d;
        margin-top: 0.75rem;
    }
    .status-bar.empty {
        background: #fefce8;
        border-color: #fef08a;
        color: #a16207;
    }
    
    /* Source tag */
    .source-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Chat styling */
    .chat-answer {
        line-height: 1.7;
        color: #1a1a1a;
    }
    
    /* Sidebar clean */
    [data-testid="stSidebar"] {
        background-color: #fff;
        border-right: 1px solid #eee;
    }
    
    .sidebar-section {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #999;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .file-item {
        padding: 0.35rem 0.5rem;
        background: #f8f8f8;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #333;
        margin-bottom: 0.25rem;
    }
    
    /* Nicer buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
    }
    
    /* Chunk detail */
    .chunk-meta {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-section">Documents</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Add files", 
        type=['pdf', 'txt', 'md'], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for f in uploaded_files:
            st.markdown(f'<div class="file-item">📄 {f.name}</div>', unsafe_allow_html=True)
        
        if st.button("Index Documents", type="primary", use_container_width=True):
            file_paths = []
            for file in uploaded_files:
                file_path = os.path.join("data", file.name)
                with open(file_path, "wb") as f_out:
                    f_out.write(file.getbuffer())
                file_paths.append(file_path)
            
            with st.spinner("Indexing..."):
                num_chunks = pipeline.ingest_files(file_paths)
                st.success(f"Done — {num_chunks} chunks indexed.")
                st.session_state['indexed'] = True
                # Clear indexed documents (only show after indexing)
                if st.session_state.get('indexed'):
                    st.markdown('<div class="sidebar-section">Management</div>', unsafe_allow_html=True)
                    if st.button("Clear Indexed Documents", type="secondary"):
                        confirm = st.warning("⚠️ This will delete all indexed data. Continue?")
                        if st.button("Yes, clear", key="clear_confirm"):
                            pipeline.clear_index()
                            st.success("✅ Index cleared. You can now upload new documents.")
                            st.session_state['indexed'] = False
                            st.session_state.messages = []
                            if "index_summary" in st.session_state:
                                del st.session_state["index_summary"]
    
    st.divider()
    st.markdown('<div class="sidebar-section">Search Settings</div>', unsafe_allow_html=True)
    distance_threshold = st.slider(
        "Strictness", 
        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="Lower = stricter matching. Higher = more lenient."
    )
    debug_mode = st.toggle("Show retrieved chunks")

# --- Main Content ---

# Header
db_count = pipeline.get_document_count()

st.markdown(f"""
<div class="app-header">
    <h1>📎 RAG Assistant</h1>
    <p>Ask questions — answers come only from your documents.</p>
    <div class="status-bar {'empty' if db_count == 0 else ''}">
        {'⚠️ No documents indexed yet' if db_count == 0 else f'✓ {db_count} chunks ready'}
    </div>
</div>
""", unsafe_allow_html=True)

# Example question buttons
example_questions = [
    "Summarize this document",
    "What are the key points?",
    "What tasks are mentioned?",
    "What deadline is mentioned?",
    "What does this document say about the project?",
]
cols = st.columns(len(example_questions))
for col, q in zip(cols, example_questions):
    if col.button(q, key=f"example_{q.replace(' ', '_')}"):
        # Simulate query handling
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = pipeline.ask(query=q, distance_threshold=distance_threshold)
                answer = result["answer"]
                chunks = result["context_chunks"]
                # Confidence label
                if chunks:
                    best_dist = min(c["distance"] for c in chunks)
                    label, color = format_confidence(best_dist)
                    st.markdown(f"<div class='confidence' style='color:{color}'>{label}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='confidence' style='color:#9e9e9e'>Insufficient evidence</div>", unsafe_allow_html=True)
                if answer.strip() == "I could not find enough information in the uploaded documents.":
                    st.warning(answer)
                    sources = []
                else:
                    st.markdown(f"<div class='card'><p>{answer}</p></div>", unsafe_allow_html=True)
                    sources = list({c["metadata"].get("filename", "Unknown") for c in chunks})
                    if sources:
                        src_html = " ".join([f'<span class="source-badge">📄 {s}</span>' for s in sources])
                        st.markdown(f"**Sources:** {src_html}", unsafe_allow_html=True)
                if debug_mode and chunks:
                    with st.expander("Evidence Used"):
                        for i, ch in enumerate(chunks, 1):
                            fname = ch["metadata"].get("filename", "Unknown")
                            dist = ch["distance"]
                            st.markdown(f"*Chunk {i} – {fname} – distance: {dist:.4f}*")
                            st.text(ch["text"][:500])
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "answer": answer,
            "sources": sources if chunks else [],
            "chunks": chunks,
        })
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            # Assistant message with answer + sources
            st.markdown(f'<div class="chat-answer">{msg["answer"]}</div>', unsafe_allow_html=True)
            
            if msg.get("sources"):
                st.markdown("---")
                sources_html = " ".join([f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]])
                st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)
            
            # Debug chunks inside the message
            if msg.get("chunks") and debug_mode:
                with st.expander("Retrieved chunks"):
                    for i, chunk in enumerate(msg["chunks"]):
                        filename = chunk['metadata'].get('filename', 'Unknown')
                        dist = chunk['distance']
                        st.markdown(f'<div class="chunk-meta">Chunk {i+1} · {filename} · distance: {dist:.4f}</div>', unsafe_allow_html=True)
                        st.text(chunk['text'][:500])
                        if i < len(msg["chunks"]) - 1:
                            st.divider()

# Chat input
if query := st.chat_input("Ask something from your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Reading your documents..."):
            result = pipeline.ask(query=query, distance_threshold=distance_threshold)
            answer = result["answer"]
            chunks = result["context_chunks"]
            
            fallback = "I could not find enough information in the uploaded documents."
            
            if fallback in answer:
                st.warning(answer)
                sources = []
            else:
                st.markdown(f'<div class="chat-answer">{answer}</div>', unsafe_allow_html=True)
                sources = list(set(c['metadata'].get('filename', 'Unknown') for c in chunks))
                
                if sources:
                    st.markdown("---")
                    sources_html = " ".join([f'<span class="source-tag">📄 {s}</span>' for s in sources])
                    st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)
            
            if chunks and debug_mode:
                with st.expander("Retrieved chunks"):
                    for i, chunk in enumerate(chunks):
                        filename = chunk['metadata'].get('filename', 'Unknown')
                        dist = chunk['distance']
                        st.markdown(f'<div class="chunk-meta">Chunk {i+1} · {filename} · distance: {dist:.4f}</div>', unsafe_allow_html=True)
                        st.text(chunk['text'][:500])
                        if i < len(chunks) - 1:
                            st.divider()
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "answer": answer,
                "sources": sources,
                "chunks": chunks
            })
