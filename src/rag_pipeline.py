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

    def get_document_count(self) -> int:
        """Returns the number of chunks currently stored in the vector database."""
        return self.vector_store.collection.count()

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

    def ask(self, query: str, top_k: int = 5, distance_threshold: float = 1.0) -> dict:
        """
        Retrieves relevant context and generates an answer.
        distance_threshold of 1.0 is a reasonable default for cosine distance 
        (values closer to 0 are better).
        Returns the answer and the retrieved chunks used for context.
        """
        # 0. Handle casual / conversational messages without hitting retrieval
        casual_reply = self._is_casual_message(query)
        if casual_reply:
            return {"answer": casual_reply, "context_chunks": []}

        # 1. Retrieve chunks
        retrieved_chunks = self.vector_store.retrieve(
            query=query, 
            top_k=top_k, 
            distance_threshold=distance_threshold
        )
        
        # 2. Generate answer
        answer = self.generator.generate_answer(query, retrieved_chunks)
        
        return {
            "answer": answer,
            "context_chunks": retrieved_chunks
        }
