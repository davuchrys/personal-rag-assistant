from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Splits text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    This splitter is smarter than a simple character-based approach because it 
    tries to split text at natural boundaries in this priority order:
    1. Paragraphs (\\n\\n)
    2. Lines (\\n)
    3. Sentences (. ! ?)
    4. Words (spaces)
    5. Characters (last resort)
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
        )

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Takes a list of document dicts (with 'text' and 'metadata')
        and returns a list of chunk dicts.
        """
        chunked_docs = []
        for doc in documents:
            text = doc['text']
            metadata = doc['metadata']
            
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                if chunk.strip():
                    # We create a shallow copy of metadata so we don't accidentally mutate
                    chunk_meta = dict(metadata)
                    chunked_docs.append({
                        "text": chunk,
                        "metadata": chunk_meta
                    })
                    
        return chunked_docs
