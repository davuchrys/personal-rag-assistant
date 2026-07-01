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

    def _generate_with_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501", # Optional, for OpenRouter rankings
            "X-Title": "Personal RAG Assistant" # Optional, for OpenRouter rankings
        }
        payload = {
            "model": "google/gemma-7b-it:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}: {response.text}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"OpenRouter API error: {e}")

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """
        Generates an answer from the context chunks.
        If no chunks are provided, returns the fallback response.
        Includes a quality gate to reject low-relevance chunks.
        """
        fallback_response = "I could not find enough information in the uploaded documents."
        
        if not context_chunks:
            return fallback_response
        
        # QUALITY GATE: If the best chunk is still too distant, refuse to answer.
        # This prevents the LLM from trying to answer with irrelevant context.
        best_distance = min(c.get("distance", 999) for c in context_chunks)
        if best_distance > 0.75:
            return fallback_response
            
        # Build the context string
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            filename = chunk['metadata'].get('filename', 'Unknown Source')
            text = chunk['text']
            context_text += f"\n--- Source {i+1}: {filename} ---\n{text}\n"

        system_prompt = """You are a strict document-grounded assistant. You MUST follow these rules with ZERO exceptions:

RULE 1 — ONLY USE THE CONTEXT BELOW.
You may ONLY use information that is explicitly stated in the Context Documents provided by the user. Do NOT use your own training data, general knowledge, or common sense to fill in gaps.

RULE 2 — NEVER INVENT OR ASSUME.
If a fact, number, name, date, or detail is NOT explicitly written in the Context Documents, you MUST NOT mention it, guess it, or infer it. Do not say "likely", "probably", or "it is possible".

RULE 3 — ADMIT WHEN YOU DON'T KNOW.
If the Context Documents do not contain enough information to fully answer the question, you MUST respond with EXACTLY this sentence and nothing else:
"I could not find enough information in the uploaded documents."

RULE 4 — NO SOURCE FILENAMES.
Do NOT mention source filenames in your answer. Sources are displayed separately by the app.

RULE 5 — BE COMPREHENSIVE AND DETAILED.
When the context DOES contain the answer, provide a thorough, well-structured response:
- Extract ALL relevant details from the context, not just the first match.
- Use bullet points, numbered lists, or paragraphs to organize information clearly.
- Explain concepts, relationships, and key details as if teaching someone the topic.
- If the context contains definitions, examples, comparisons, or statistics, include them.
- Aim for a complete answer that covers every aspect mentioned in the context documents."""

        user_prompt = f"""Context Documents:
{context_text}

Question:
{query}

Remember: If the answer is not in the Context Documents above, say "I could not find enough information in the uploaded documents." — do NOT make anything up."""
        
        try:
            from dotenv import dotenv_values
            env_vars = dotenv_values(".env")
            use_ollama_str = env_vars.get("USE_OLLAMA") or os.getenv("USE_OLLAMA", "false")
            use_ollama = str(use_ollama_str).strip().lower() == "true"
            
            if use_ollama:
                # Ollama uses a single prompt, so combine system + user
                full_prompt = system_prompt + "\n\n" + user_prompt
                answer = self._generate_with_ollama(full_prompt)
            else:
                answer = self._generate_with_openrouter(system_prompt, user_prompt)
                
            if not answer:
                return fallback_response
            return answer
                
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"An error occurred while generating the answer: {e}"

