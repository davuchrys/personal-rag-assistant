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

    def ask(self, query: str, top_k: int = 5, distance_threshold: float = 1.0) -> dict:
        """
        Retrieves relevant context and generates an answer.
        distance_threshold of 1.0 is a reasonable default for cosine distance 
        (values closer to 0 are better).
        Returns the answer and the retrieved chunks used for context.
        """
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
