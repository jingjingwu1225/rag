"""
agent_query.py
Agentic, memory-aware RAG CLI.

Unlike query.py (one fixed retrieve -> generate pass), every question here
goes through the LangGraph agent in agent_graph.py, which:
  1. retrieves chunks
  2. grades whether they're actually sufficient to answer the question
  3. if not, rewrites the search query and retries (up to MAX_RETRIES times)
  4. generates the final answer, using the whole conversation as context

Run `python ingest.py` first to build the knowledge base, then:

    python agent_query.py
"""

import uuid

from agent_graph import prepare_turn, stream_answer
from rag_core import summarize_sources


def main():
    print("Agentic RAG Q&A over your papers (self-correcting + memory). Type 'quit' to exit.\n")
    thread_id = str(uuid.uuid4())  # one persisted conversation per CLI session

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        state = prepare_turn(question, thread_id=thread_id)

        if state.get("retry_count", 0) > 0:
            print(
                f"  (retrieved context was weak — query rewritten "
                f"{state['retry_count']}x -> \"{state['search_query']}\")"
            )
        if state.get("grade_reason"):
            print(f"  (grader: {state['grade_reason']}, confidence={state.get('grade_confidence')})")

        print("\nAnswer:")
        for token in stream_answer(question, state, thread_id=thread_id):
            print(token, end="", flush=True)
        print("\n")

        print("Sources used:")
        for s in summarize_sources(state["retrieved_chunks"]):
            passages = f"{s['count']} passage" + ("s" if s["count"] != 1 else "")
            if s["rerank_scores"]:
                detail = f"rerank scores {sorted(s['rerank_scores'], reverse=True)}"
            else:
                detail = f"distances {[round(d, 3) for d in s['distances']]}"
            print(f"  - {s['source']} — {passages} ({detail})")
        print()


if __name__ == "__main__":
    main()
