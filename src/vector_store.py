import os
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    """Manages the vector database using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./vector_db", collection_name: str = "rag_collection"):
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize chroma client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # We explicitly use Sentence Transformers
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity space
        )

    def get_indexed_files(self) -> list[str]:
        """Returns a list of unique filenames currently indexed."""
        if self.collection.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        if not data or not data["metadatas"]:
            return []
        
        files = set()
        for meta in data["metadatas"]:
            if meta and "filename" in meta:
                files.add(meta["filename"])
        return sorted(list(files))

    def add_chunks(self, chunks: list[dict]):
        """Adds text chunks and their metadata to the vector store."""
        if not chunks:
            return
            
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk['text'])
            metadatas.append(chunk['metadata'])
            # Generate a unique ID based on the filename and index
            filename = chunk['metadata'].get('filename', 'unknown')
            ids.append(f"{filename}_{i}_{hash(chunk['text'])}")
            
        # Add to chroma DB
        # We can add in batches if the list is huge, but assuming reasonable size here
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve(self, query: str, top_k: int = 5, distance_threshold: float = 0.5) -> list[dict]:
        """
        Retrieves the most relevant chunks for a query.
        Lower distance means higher similarity in cosine space (0 is identical, 1 is orthogonal).
        distance_threshold specifies the maximum acceptable distance.
        """
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved_chunks = []
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            # distance_threshold filtering
            if dist <= distance_threshold:
                retrieved_chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
                
        return retrieved_chunks
