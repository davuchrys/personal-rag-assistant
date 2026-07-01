import os
import requests


OLLAMA_MODEL = "llama3" # You can also use "phi3" or "mistral"

class AnswerGenerator:
    """Generates answers using Groq API (Cloud) or local Ollama based on retrieved context."""
    
    def __init__(self):
        pass

    def _generate_with_ollama(self, prompt: str) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0 # Low temperature for factual retrieval
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise Exception(f"Ollama error: {e}. Make sure Ollama is installed and running!")

    def _generate_with_groq(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}: {response.text}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Groq API error: {e}")

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
            from dotenv import dotenv_values
            env_vars = dotenv_values(".env")
            use_ollama_str = env_vars.get("USE_OLLAMA") or os.getenv("USE_OLLAMA", "false")
            use_ollama = str(use_ollama_str).strip().lower() == "true"
            
            if use_ollama:
                answer = self._generate_with_ollama(prompt)
            else:
                answer = self._generate_with_groq(prompt)
                
            if not answer:
                return fallback_response
            return answer
                
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"An error occurred while generating the answer: {e}"
