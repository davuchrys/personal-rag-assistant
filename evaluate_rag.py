"""Offline RAG evaluation using an LLM-as-judge (no RAGAS dependency).

RAGAS was tried first but conflicts with this project's langchain_openai
version (it needs an old langchain-community that no longer ships
langchain_community.chat_models.vertexai) — forcing it in broke answer
generation entirely. This script reuses the same OpenRouter/Ollama backend
already used for answers to score faithfulness and relevancy instead.

This is a standalone dev tool, not part of the live app — run it manually
whenever you want a quality check, e.g. after changing chunking, retrieval,
or prompt logic.

Usage:
    python evaluate_rag.py --username <your_username>

Edit TEST_CASES below to match documents you've actually indexed for that
user — a question is only a fair test if the answer is supposed to be in
their vector store.
"""

import argparse
import json
import statistics
import sys

from src.rag_pipeline import RAGPipeline

# Edit this list to match documents indexed for the user you're evaluating.
# 'reference' is optional — when given, the judge also scores 'correctness'
# (does the answer capture the key fact in the reference).
TEST_CASES = [
    {
        "question": "What detection accuracy did the stacking ensemble deep learning model achieve for real-time ransomware detection in IoMT?",
        "reference": "Over 99.3% detection accuracy",
    },
    {
        "question": "How many studies were included in the final corpus after the PRISMA screening process in this review?",
        "reference": "112 studies",
    },
    {
        "question": "What is the CIANA model?",
        "reference": "Confidentiality, Integrity, Availability, Non-repudiation, Authentication — a specialized extension of the CIA triad for medical networks.",
    },
    {
        "question": "What was the median cost of healthcare data breaches in 2023 according to this survey?",
        "reference": "USD 10.93 million",
    },
    {
        # No reference on purpose: a deliberately out-of-scope question to
        # confirm the pipeline correctly refuses rather than hallucinating.
        "question": "What is the best recipe for chocolate lava cake?",
        "reference": None,
    },
]


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no context retrieved)"
    return "\n\n".join(f"[Source: {c['metadata'].get('filename', 'unknown')}] {c['text']}" for c in chunks)


def run_evaluation(username: str):
    pipeline = RAGPipeline(vector_db_path=f"./vector_db/{username}", username=username)

    results = []
    for case in TEST_CASES:
        question = case["question"]
        reference = case.get("reference")

        result = pipeline.ask(query=question, chat_history=[])
        answer = result["answer"]
        chunks = result["context_chunks"]
        context_text = format_context(chunks)

        scores = pipeline.generator.judge_answer(question, answer, context_text, reference=reference)

        results.append({
            "question": question,
            "answer": answer,
            "reference": reference,
            "chunks_found": len(chunks),
            "retrieval_trace": result.get("retrieval_trace", []),
            "scores": scores,
        })

        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"Scores: {scores}")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    faithfulness_scores = [r["scores"].get("faithfulness") for r in results if r["scores"].get("faithfulness") is not None]
    relevancy_scores = [r["scores"].get("relevancy") for r in results if r["scores"].get("relevancy") is not None]
    correctness_scores = [r["scores"].get("correctness") for r in results if r["scores"].get("correctness") is not None]

    if faithfulness_scores:
        print(f"Avg faithfulness: {statistics.mean(faithfulness_scores):.2f} (n={len(faithfulness_scores)})")
    if relevancy_scores:
        print(f"Avg relevancy:    {statistics.mean(relevancy_scores):.2f} (n={len(relevancy_scores)})")
    if correctness_scores:
        print(f"Avg correctness:  {statistics.mean(correctness_scores):.2f} (n={len(correctness_scores)})")

    report_path = f"data/logs/eval_report_{username}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline's answer quality with an LLM judge.")
    parser.add_argument("--username", required=True, help="Username whose vector_db to evaluate against.")
    args = parser.parse_args()
    run_evaluation(args.username)
