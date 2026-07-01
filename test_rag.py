import os
from dotenv import load_dotenv
load_dotenv()
from src.rag_pipeline import RAGPipeline

def main():
    print("Testing RAG Pipeline...")
    pipeline = RAGPipeline(vector_db_path="./test_vector_db")
    
    print("Ingesting test.txt...")
    chunks_added = pipeline.ingest_files(["test.txt"])
    print(f"Chunks added: {chunks_added}")
    
    print("Querying: 'What is the capital of France?'")
    result = pipeline.ask("What is the capital of France?", top_k=2, distance_threshold=1.5)
    
    print(f"Answer: {result['answer']}")
    print(f"Context chunks used: {len(result['context_chunks'])}")
    for i, c in enumerate(result['context_chunks']):
        print(f"  Chunk {i}: {c['metadata']['filename']} (dist: {c['distance']:.3f})")
        
    print("\nQuerying missing info: 'Who is the CEO of Google?'")
    result2 = pipeline.ask("Who is the CEO of Google?", top_k=2, distance_threshold=1.5)
    print(f"Answer: {result2['answer']}")

if __name__ == "__main__":
    main()
