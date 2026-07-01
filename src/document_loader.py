import os
import pypdf

class DocumentLoader:
    """Loads documents from various file types."""
    
    @staticmethod
    def load_document(file_path: str) -> list[dict]:
        """
        Loads a document and returns a list of dictionaries.
        Each dictionary contains 'text' and 'metadata'.
        """
        _, ext = os.path.splitext(file_path.lower())
        
        filename = os.path.basename(file_path)
        metadata = {"filename": filename, "source": file_path}
        
        if ext == '.pdf':
            return DocumentLoader._load_pdf(file_path, metadata)
        elif ext in ['.txt', '.md']:
            return DocumentLoader._load_text(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    @staticmethod
    def _load_pdf(file_path: str, metadata: dict) -> list[dict]:
        docs = []
        try:
            with open(file_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                if full_text.strip():
                    docs.append({
                        "text": full_text,
                        "metadata": metadata
                    })
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            
        return docs

    @staticmethod
    def _load_text(file_path: str, metadata: dict) -> list[dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                return [{
                    "text": text,
                    "metadata": metadata
                }]
        except Exception as e:
            print(f"Error loading Text/Markdown {file_path}: {e}")
            return []
