import os
from google import genai
from google.genai import types

class AnswerGenerator:
    """Generates answers using Gemini API based on retrieved context."""
    
    def __init__(self):
        # Initialize Google GenAI client (it reads GEMINI_API_KEY from environment by default)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-1.5-flash'  # Higher free-tier quota (1500 req/day vs 20)

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """
        Generates an answer from the context chunks.
        If no chunks are provided, returns the fallback response.
        """
        fallback_response = "I could not find enough information in the uploaded documents."
        
        if not context_chunks:
            return fallback_response
            
        # Build the context string
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            filename = chunk['metadata'].get('filename', 'Unknown Source')
            text = chunk['text']
            context_text += f"\n--- Source {i+1}: {filename} ---\n{text}\n"

        prompt = f"""You are a helpful personal assistant answering questions strictly based on the provided context documents.

Context Documents:
{context_text}

Question:
{query}

Instructions:
1. Answer the question using ONLY the information from the Context Documents above.
2. Do NOT include source filenames in your answer. Sources are displayed separately.
3. Write clear, natural answers as if you are explaining to a person.
4. If the Context Documents do not contain enough information or are unrelated to the question, you MUST return EXACTLY this sentence:
"I could not find enough information in the uploaded documents."
Do not add anything else if the information is missing.
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0 # Low temperature for factual retrieval
                )
            )
            
            # Additional safety: if the model tries to answer but we know context was weak, 
            # though the threshold handles weak context, this ensures strict compliance.
            if not response.text:
                return fallback_response
                
            return response.text
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"An error occurred while generating the answer: {e}"
