import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables early so the pipeline can access GEMINI_API_KEY
load_dotenv()

from src.rag_pipeline import RAGPipeline

# --- Setup ---
st.set_page_config(page_title="Personal RAG Assistant", layout="wide")

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# --- UI Layout ---
st.title("🧠 Personal RAG Assistant")

# --- Sidebar for Uploading and Indexing ---
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=['pdf', 'txt', 'md'], 
        accept_multiple_files=True
    )
    
    if st.button("Index Documents"):
        if uploaded_files:
            file_paths = []
            for file in uploaded_files:
                # Save uploaded file to local disk
                file_path = os.path.join("data", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)
            
            with st.spinner("Indexing documents..."):
                num_chunks = pipeline.ingest_files(file_paths)
                st.success(f"Successfully indexed {len(uploaded_files)} files ({num_chunks} chunks).")
        else:
            st.warning("Please upload files first.")
            
    st.divider()
    st.header("Settings")
    debug_mode = st.checkbox("Debug Mode (Show Retrieved Chunks)")
    distance_threshold = st.slider("Distance Threshold", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                   help="Lower value means stricter matching (fewer chunks). Higher value allows weaker matches.")

# --- Main Area for QA ---
st.header("Ask a Question")

query = st.text_input("Enter your question based on the uploaded documents:")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        result = pipeline.ask(query=query, distance_threshold=distance_threshold)
        answer = result["answer"]
        chunks = result["context_chunks"]
        
        st.subheader("Answer")
        st.write(answer)
        
        if debug_mode:
            st.divider()
            st.subheader("Debug: Retrieved Chunks")
            if not chunks:
                st.info("No chunks were retrieved (either vector DB is empty or all chunks exceeded the distance threshold).")
            else:
                for i, chunk in enumerate(chunks):
                    with st.expander(f"Chunk {i+1} (Source: {chunk['metadata'].get('filename', 'Unknown')}) | Distance: {chunk['distance']:.4f}"):
                        st.text(chunk['text'])
