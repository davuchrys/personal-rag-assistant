class TextChunker:
    """Splits text into chunks with overlap."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Takes a list of document dicts (with 'text' and 'metadata')
        and returns a list of chunk dicts.
        """
        chunked_docs = []
        for doc in documents:
            text = doc['text']
            metadata = doc['metadata']
            
            chunks = self._split_text(text)
            for chunk in chunks:
                if chunk.strip():
                    # We create a shallow copy of metadata so we don't accidentally mutate
                    chunk_meta = dict(metadata)
                    chunked_docs.append({
                        "text": chunk,
                        "metadata": chunk_meta
                    })
                    
        return chunked_docs

    def _split_text(self, text: str) -> list[str]:
        """Splits a single string into chunks with overlap."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to find a nice break point
            if end < text_length:
                # Try to break at a newline or space within the last 50 characters
                break_point = max(text.rfind('\n', start, end), text.rfind(' ', start, end))
                # If we found a suitable break point, adjust the end
                if break_point != -1 and break_point > start + self.chunk_size - 100:
                    end = break_point + 1 # Include the space/newline
            
            chunks.append(text[start:end])
            
            # Calculate next start point taking overlap into account
            if end >= text_length:
                break
            
            start = end - self.overlap
            # Avoid getting stuck if overlap is too large
            if start <= 0 or end - start <= 0:
                start = end
                
        return chunks
