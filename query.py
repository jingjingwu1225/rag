"""
query.py
A simple command-line loop for asking questions against your knowledge base.
Run `python ingest.py` first to build the knowledge base, then:

    python query.py
"""

from rag_core import retrieve_reranked, generate_answer_stream, summarize_sources


def main():
    print("RAG Q&A over your papers. Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        retrieved_chunks = retrieve_reranked(question, k=4)
        if not retrieved_chunks:
            print("No relevant chunks found — did you run ingest.py?\n")
            continue

        print("\nAnswer:")
        for token in generate_answer_stream(question, retrieved_chunks):
            print(token, end="", flush=True)
        print("\n")

        print("Sources used:")
        for s in summarize_sources(retrieved_chunks):
            passages = f"{s['count']} passage" + ("s" if s["count"] != 1 else "")
            if s["rerank_scores"]:
                detail = f"rerank scores {sorted(s['rerank_scores'], reverse=True)}"
            else:
                detail = f"distances {[round(d, 3) for d in s['distances']]}"
            print(f"  - {s['source']} — {passages} ({detail})")
        print()


if __name__ == "__main__":
    main()
