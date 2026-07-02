import os
import requests


OLLAMA_MODEL = "llama3" # You can also use "phi3" or "mistral"

class AnswerGenerator:
    """Generates answers using OpenRouter (via LangChain) or local Ollama based on retrieved context."""
    
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
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")
            
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model="openrouter/free",
                temperature=0.0,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Personal RAG Assistant"
                }
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"LangChain OpenRouter error: {e}")

    def _build_prompt(self, query: str, context_text: str, chat_history: list[dict] = None) -> tuple[str, str]:
        """Build system and user prompts using ChatPromptTemplate style.
        
        Incorporates conversation history so the model can understand
        follow-up questions like "explain more" or "give me examples".
        """
        from langchain_core.prompts import ChatPromptTemplate

        system_template = """You are a strict document-grounded assistant. You MUST follow these rules with ZERO exceptions:

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
- Aim for a complete answer that covers every aspect mentioned in the context documents.

RULE 6 — USE CONVERSATION HISTORY FOR CONTEXT.
If the user asks a follow-up question (e.g. "explain more", "give examples", "what about X?"), use the Conversation History below to understand what they are referring to. Still ONLY answer from the Context Documents."""

        # Build conversation history string
        history_text = ""
        if chat_history:
            # Keep only last 5 exchanges to avoid token overflow
            recent = chat_history[-10:]  # 10 messages = ~5 exchanges (user+assistant)
            for msg in recent:
                role = msg.get("role", "")
                if role == "user":
                    history_text += f"\nUser: {msg.get('content', '')}"
                elif role == "assistant":
                    history_text += f"\nAssistant: {msg.get('answer', '')}"
        
        if history_text:
            user_content = f"""Conversation History:
{history_text}

Context Documents:
{context_text}

Current Question:
{query}

Remember: If the answer is not in the Context Documents above, say "I could not find enough information in the uploaded documents." — do NOT make anything up."""
        else:
            user_content = f"""Context Documents:
{context_text}

Question:
{query}

Remember: If the answer is not in the Context Documents above, say "I could not find enough information in the uploaded documents." — do NOT make anything up."""

        return system_template, user_content

    def generate_answer(self, query: str, context_chunks: list[dict], chat_history: list[dict] = None) -> str:
        """
        Generates an answer from the context chunks.
        If no chunks are provided, returns the fallback response.
        Includes a quality gate to reject low-relevance chunks.
        
        Args:
            query: The user's question.
            context_chunks: List of retrieved document chunks.
            chat_history: Optional list of previous messages for conversation memory.
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

        # Build prompts with conversation history
        system_prompt, user_prompt = self._build_prompt(query, context_text, chat_history)
        
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
