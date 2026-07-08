import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# Default kept identical to the original hardcoded model so existing deployed
# indexes keep working unless EMBEDDING_MODEL is explicitly set. Switching
# models changes the vector space, so any existing collection must be cleared
# (see app.py "Clear Indexed Data") and re-indexed after changing this.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class VectorStore:
    """Manages the vector database using ChromaDB."""

    def __init__(self, persist_directory: str = "./vector_db", collection_name: str = "rag_collection", embedding_model: str = None):
        # Ensure the directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize chroma client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Embedding model is configurable (constructor arg > EMBEDDING_MODEL env
        # var > default) so a stronger model can be evaluated without touching
        # code, while defaulting to the original model for backward compatibility.
        model_name = embedding_model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity space
        )

        # Optional cross-encoder reranker: off by default because it loads a
        # second transformer model, adding real memory/startup cost — not
        # something to switch on automatically for an already-live deployment
        # on a resource-constrained host. Enable via ENABLE_RERANKER=true.
        self._reranker = None
        if os.getenv("ENABLE_RERANKER", "false").strip().lower() == "true":
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

    def delete_by_filename(self, filename: str) -> None:
        """Removes all chunks previously indexed for a given filename.

        Called before re-ingesting a file with the same name so re-uploading
        a document replaces its chunks instead of duplicating them.
        """
        if self.collection.count() == 0:
            return
        self.collection.delete(where={"filename": filename})

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

    def _bm25_rerank(self, query: str, candidates: list[dict], alpha: float = 0.5) -> list[dict]:
        """Re-ranks vector-search candidates by blending in a BM25 keyword score.

        Pure semantic search can miss exact keyword/acronym matches (e.g. an
        error code or a proper noun); BM25 catches those. The BM25 index is
        built fresh from just this query's candidate set rather than the whole
        collection, so there's no separate index to keep in sync with add/delete.
        """
        if len(candidates) <= 1:
            return candidates

        tokenized_corpus = [c["text"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())

        max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0

        scored = []
        for candidate, bm25_score in zip(candidates, bm25_scores):
            vector_similarity = 1 - candidate["distance"]
            normalized_bm25 = bm25_score / max_bm25
            combined = alpha * vector_similarity + (1 - alpha) * normalized_bm25
            scored.append((combined, candidate))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [candidate for _, candidate in scored]

    def _cross_encoder_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Optional second-stage rerank with a cross-encoder (only if ENABLE_RERANKER=true)."""
        if not self._reranker or len(candidates) <= 1:
            return candidates
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["_rerank_score"] = float(score)
        candidates.sort(key=lambda c: c["_rerank_score"], reverse=True)
        return candidates

    def retrieve(self, query: str, top_k: int = 5, distance_threshold: float = 0.5, use_hybrid: bool = True) -> list[dict]:
        """
        Retrieves the most relevant chunks for a query.
        Lower distance means higher similarity in cosine space (0 is identical, 1 is orthogonal).
        distance_threshold specifies the maximum acceptable distance.

        When use_hybrid is True, casts a wider net via vector search, then
        re-ranks those candidates with a BM25 keyword score (and, if enabled,
        a cross-encoder) before applying the distance threshold and top_k cutoff.
        The original cosine distance is preserved on every candidate, so the
        distance-based quality gate downstream (see generator.py) is unaffected.
        """
        if self.collection.count() == 0:
            return []

        candidate_k = min(top_k * 4, self.collection.count()) if use_hybrid else top_k
        results = self.collection.query(
            query_texts=[query],
            n_results=candidate_k
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        candidates = [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
        ]

        if use_hybrid:
            candidates = self._bm25_rerank(query, candidates)
            candidates = self._cross_encoder_rerank(query, candidates)

        # distance_threshold filtering, then cap to top_k in (re-)ranked order
        retrieved_chunks = [c for c in candidates if c["distance"] <= distance_threshold][:top_k]
        return retrieved_chunks
