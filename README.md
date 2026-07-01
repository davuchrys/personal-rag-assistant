# Personal RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant that lets you talk to your documents locally.

## Features
- Upload PDF, TXT, and Markdown files.
- Local vector search using ChromaDB and SentenceTransformers.
- Question answering using Gemini 2.5 Flash via `google-genai`.
- Strict constraints to answer only from context and provide source citations.
- Built-in Streamlit UI with a debug mode for inspecting retrieved chunks.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Key**:
   Ensure you have a `.env` file in the project root containing your Gemini API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Use the sidebar to upload one or multiple documents.
2. Click **Index Documents** to process and store them in the local vector database.
3. Type a question in the main chat area to get an answer based on your documents.
4. Toggle **Debug Mode** in the sidebar to see exactly which text chunks were retrieved for the query.
