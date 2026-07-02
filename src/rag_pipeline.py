import os
from src.document_loader import DocumentLoader
from src.chunker import TextChunker
from src.vector_store import VectorStore
from src.generator import AnswerGenerator

class RAGPipeline:
    """Orchestrates the entire RAG workflow."""
    
    def __init__(self, vector_db_path: str = "./vector_db"):
        self.chunker = TextChunker(chunk_size=1000, overlap=200)
        self.vector_store = VectorStore(persist_directory=vector_db_path)
        self.generator = AnswerGenerator()
        # Store path for clearing later
        self.persist_directory = vector_db_path

    def get_document_count(self) -> int:
        """Returns the number of chunks currently stored in the vector database."""
        return self.vector_store.collection.count()

    def get_indexed_files(self) -> list[str]:
        """Returns a list of unique filenames currently indexed."""
        return self.vector_store.get_indexed_files()

    def ingest_files(self, file_paths: list[str]) -> int:
        """
        Loads, chunks, and stores files in the vector database.
        Returns the number of chunks added.
        """

        all_chunks = []
        
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            # 1. Load document
            docs = DocumentLoader.load_document(file_path)
            
            # 2. Chunk document
            chunks = self.chunker.chunk_documents(docs)
            all_chunks.extend(chunks)
            
        # 3. Store chunks
        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
            
        return len(all_chunks)

    # Simple patterns for casual/conversational messages
    GREETING_PATTERNS = {
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "howdy", "sup", "what's up", "whats up", "yo",
    }
    THANKS_PATTERNS = {
        "thanks", "thank you", "thx", "ty", "appreciate it",
    }

    def _is_casual_message(self, query: str) -> str | None:
        """Returns a friendly reply if the query is casual chit-chat, else None."""
        q = query.strip().lower().rstrip("!?.,")
        
        if q in self.GREETING_PATTERNS:
            return "Hey! I'm ready to help. Ask me anything about your uploaded documents."
        if q in self.THANKS_PATTERNS:
            return "You're welcome! Let me know if you have more questions."
        if q in ("help", "what can you do", "what do you do"):
            return ("I can answer questions based on the documents you've uploaded. "
                    "Just upload a file in the sidebar, index it, and ask away!")
        return None

    def clear_index(self):
        """Deletes the persistent vector store and reinitializes the collection.
        Used by the UI to reset indexed documents.
        """
        import shutil
        # Remove the persistence directory if it exists
        if os.path.isdir(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        # Re‑create a fresh VectorStore instance
        self.vector_store = VectorStore(persist_directory=self.persist_directory)
        # Reset any session state flags that indicate indexing has occurred
        try:
            import streamlit as st
            for key in ["indexed", "messages"]:
                if key in st.session_state:
                    st.session_state.pop(key)
        except Exception:
            pass

    def _reformulate_query(self, query: str, chat_history: list[dict]) -> str:
        """Rewrite a vague follow-up question into a standalone, specific query.
        
        Uses LangChain + OpenRouter to understand conversation context and 
        produce a query that will match relevant document chunks in vector search.
        
        Example:
            History: "What is IDS?" → [answer about IDS]
            Query: "explain it more detail"
            Output: "Explain IDS intrusion detection system types capabilities and evaluation metrics in detail"
        """
        if not chat_history:
            return query
        
        # Build recent history (last 6 messages = ~3 exchanges)
        recent = chat_history[-6:]
        history_text = ""
        for msg in recent:
            role = msg.get("role", "")
            if role == "user":
                history_text += f"\nUser: {msg.get('content', '')}"
            elif role == "assistant":
                history_text += f"\nAssistant: {msg.get('answer', '')[:200]}"
        
        if not history_text.strip():
            return query
        
        try:
            from dotenv import dotenv_values
            env_vars = dotenv_values(".env")
            use_ollama_str = env_vars.get("USE_OLLAMA") or os.getenv("USE_OLLAMA", "false")
            use_ollama = str(use_ollama_str).strip().lower() == "true"
            
            reformulation_prompt = f"""Given the conversation history below, rewrite the user's latest question into a standalone, specific search query. The query should contain the key topics being discussed so it can find relevant documents.

Conversation History:
{history_text}

Latest Question: {query}

Rules:
- Output ONLY the rewritten query, nothing else.
- Keep it concise (under 30 words).
- Include the specific topic/subject from the conversation.
- If the question is already specific enough, return it unchanged.

Rewritten Query:"""

            if use_ollama:
                import requests
                url = "http://localhost:11434/api/generate"
                payload = {"model": "llama3", "prompt": reformulation_prompt, "stream": False, "options": {"temperature": 0.0}}
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                reformulated = response.json().get("response", "").strip()
            else:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage
                
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    try:
                        import streamlit as st
                        api_key = st.secrets.get("OPENROUTER_API_KEY")
                    except Exception:
                        pass
                
                if not api_key:
                    print("[Query Reformulation] Missing OPENROUTER_API_KEY")
                    return query
                    
                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="openrouter/free",
                    temperature=0.0,
                    default_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "Personal RAG Assistant"
                    }
                )
                response = llm.invoke([HumanMessage(content=reformulation_prompt)])
                reformulated = response.content.strip()
            
            if reformulated and len(reformulated) < 200:
                print(f"[Query Reformulation] '{query}' -> '{reformulated}'")
                return reformulated
            return query
            
        except Exception as e:
            import traceback
            print(f"[Query Reformulation] Failed: {e}")
            traceback.print_exc()
            return query

    def ask(self, query: str, top_k: int = 8, distance_threshold: float = 0.7, chat_history: list[dict] = None) -> dict:
        """
        Retrieves relevant context and generates an answer.
        distance_threshold of 0.7 is a strict default for cosine distance 
        (values closer to 0 are better). Chunks above this are discarded.
        Returns the answer and the retrieved chunks used for context.
        
        Args:
            chat_history: Optional list of previous messages for conversation memory.
        """
        # 0. Handle casual / conversational messages without hitting retrieval
        casual_reply = self._is_casual_message(query)
        if casual_reply:
            return {"answer": casual_reply, "context_chunks": []}

        # 1. Reformulate vague follow-up queries into specific standalone queries
        search_query = query
        if chat_history and len(chat_history) >= 2:
            search_query = self._reformulate_query(query, chat_history)

        # 2. Retrieve chunks using the (potentially reformulated) query
        retrieved_chunks = self.vector_store.retrieve(
            query=search_query, 
            top_k=top_k, 
            distance_threshold=distance_threshold
        )
        
        # 3. Generate answer with conversation history (using original query for natural response)
        answer = self.generator.generate_answer(query, retrieved_chunks, chat_history=chat_history)
        
        return {
            "answer": answer,
            "context_chunks": retrieved_chunks
        }
