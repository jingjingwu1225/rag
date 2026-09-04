"""
agent_graph.py
Agentic RAG built with LangGraph: a self-correcting retrieval loop, instead
of the single fixed "retrieve -> generate" pass in rag_core.py / query.py.

Graph — deciding *what to answer with*:

    contextualize --> decompose --> retrieve --[decomposed: multi-query]--> END
                                              `--[single query]--> grade --[insufficient, retries left]--> rewrite --> retrieve  (loop)
                                                                        `--[sufficient, or out of retries]--> END

- contextualize : if there's prior conversation, resolve a follow-up like
                  "what method do they use for that?" into a standalone
                  search query *before* the first retrieval attempt.
                  Without this, retrieval embeds the raw pronoun-laden
                  question and fails, even though generation is
                  history-aware — retrieval needs its own history awareness.
- decompose     : a single vector search over one query naturally biases
                  toward whichever document's terms dominate it, so a
                  cross-document question ("how does X in paper A compare
                  to Y in paper B?") tends to only retrieve one side of the
                  comparison. This node (JSON mode) checks whether the
                  question needs pulling from multiple documents/topics and,
                  if so, splits it into up to MAX_SUBQUERIES focused
                  sub-questions — one retrieval pass each, merged — instead
                  of one query trying to cover everything at once.
- retrieve      : runs retrieve_reranked() once per sub-query (just one, for
                  a normal question) and merges the results, deduped by
                  chunk id (rag_core.retrieve_reranked, which itself does a
                  wide hybrid vector+BM25 search reranked down to top-k)
- grade         : LLM judges (via JSON-mode structured output, not text
                  matching) whether the retrieved chunks actually answer the
                  question, with a confidence + reason — a form of
                  "Corrective RAG" / self-RAG
- rewrite       : if not, LLM rewrites the search query (also JSON mode,
                  also history-aware, and told *why* the last attempt fell
                  short) and we retry — bounded by MAX_RETRIES so a stubborn
                  question can't loop forever

Generation lives *outside* the graph on purpose: LangGraph's `.invoke()`
only returns once a run reaches END, so a node that calls the LLM has no
way to hand tokens to the caller as they arrive — the whole answer would
appear at once regardless of how it's written internally. Splitting it into
two functions fixes that without giving up memory:

  prepare_turn(question, thread_id)        -> runs the graph above, returns
                                               the finalized retrieved_chunks
                                               (does NOT call the LLM to answer)
  stream_answer(question, state, thread_id) -> streams the final answer
                                               token-by-token, then persists
                                               it (+ chat_history) back into
                                               the thread's checkpointed state

run_turn() composes both, non-streaming, for callers that just want the
final dict (eval.py, or any script that doesn't care about token-by-token
output).

Conversation memory across turns is handled by LangGraph's checkpointer
(MemorySaver): each call for the same thread_id resumes the persisted graph
state, so `chat_history` accumulates automatically instead of you having to
pass the whole conversation back in by hand — `stream_answer()` writes to
it via `AGENT.update_state()` once the streamed answer is complete.
"""

import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from rag_core import GENERATION_MODEL, OPENAI_CLIENT, generate_answer_stream, retrieve_reranked

MAX_RETRIES = 2       # cap on query-rewrite loops, so a bad question can't spin forever
MAX_SUBQUERIES = 3    # cap on decomposition, so a question can't fan out unboundedly
RETRIEVE_K = 6         # final chunk count after reranking (a bit wider than query.py's
                       # 4, since grading may still reject this pass and want a retry)


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    sub_queries: list
    retrieved_chunks: list
    grade: str
    grade_confidence: int
    grade_reason: str
    retry_count: int
    answer: str
    # Annotated with operator.add so each turn's [user, assistant] pair is
    # appended to (not overwritten on top of) whatever the checkpointer
    # already has stored for this thread_id.
    chat_history: Annotated[list, operator.add]


def _history_block(state: RAGState) -> str:
    history = state.get("chat_history")
    if not history:
        return ""
    turns = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history)
    return f"CONVERSATION HISTORY:\n{turns}\n\n"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def contextualize_node(state: RAGState) -> dict:
    """
    Turn a context-dependent follow-up ("what method do they use for
    that?") into a standalone search query, using the conversation history.
    Skips the LLM call entirely on turn 1 (no history yet) to save cost.
    """
    history_block = _history_block(state)
    if not history_block:
        return {"search_query": state["question"]}

    prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up into a standalone question that makes sense with NO prior "
        "context — resolve pronouns and implicit references (\"that\", \"it\", "
        "\"the method\", etc.) explicitly using the history. If the follow-up "
        "is already standalone, return it unchanged.\n\n"
        f"{history_block}"
        f"FOLLOW-UP QUESTION: {state['question']}\n\n"
        "Reply with ONLY the standalone question, nothing else."
    )
    response = OPENAI_CLIENT.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    standalone = (response.choices[0].message.content or "").strip()
    return {"search_query": standalone or state["question"]}


def decompose_node(state: RAGState) -> dict:
    """
    Check whether this question needs pulling from multiple documents/topics
    and synthesizing across them, and if so, split it into focused
    sub-questions — one retrieval pass per sub-question, merged, rather than
    a single query that has to represent the whole comparison at once.
    """
    query = state.get("search_query") or state["question"]
    prompt = (
        "Does answering this question require pulling information from "
        "MULTIPLE different documents or topics and synthesizing across "
        "them (e.g. a comparison, \"both X and Y\"), as opposed to a single "
        "focused lookup? If yes, split it into up to "
        f"{MAX_SUBQUERIES} focused sub-questions, one per document/aspect "
        "needed. If no, return the question itself, unchanged, as the only "
        "sub-question. Respond in JSON.\n\n"
        f"QUESTION: {query}\n\n"
        'Reply with ONLY this JSON object, no other text: '
        '{"needs_decomposition": <true or false>, "sub_queries": ["..."]}'
    )
    response = OPENAI_CLIENT.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=200,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        parsed = json.loads(response.choices[0].message.content)
        sub_queries = [q.strip() for q in (parsed.get("sub_queries") or []) if q and q.strip()]
    except (json.JSONDecodeError, TypeError):
        sub_queries = []
    return {"sub_queries": sub_queries[:MAX_SUBQUERIES] or [query]}


def retrieve_node(state: RAGState) -> dict:
    """
    Retrieve for each sub-query independently and merge, deduped by chunk
    id — for an un-decomposed question that's just one retrieval pass, same
    as before; for a decomposed one, each sub-question gets its own shot at
    surfacing chunks its terms are strong in, instead of one query diluting
    across all of them.
    """
    sub_queries = state.get("sub_queries") or [state.get("search_query") or state["question"]]
    per_query_k = RETRIEVE_K if len(sub_queries) == 1 else max(3, RETRIEVE_K // len(sub_queries) + 2)

    seen_ids = set()
    merged = []
    for sub_q in sub_queries:
        for chunk in retrieve_reranked(sub_q, k=per_query_k):
            if chunk["id"] in seen_ids:
                continue
            seen_ids.add(chunk["id"])
            merged.append({**chunk, "sub_query": sub_q})
    return {"retrieved_chunks": merged}


def grade_node(state: RAGState) -> dict:
    """
    Ask the LLM: is this retrieved context actually enough to answer?

    Uses JSON mode (response_format=json_object) instead of parsing a raw
    "SUFFICIENT"/"INSUFFICIENT" string — the earlier version's substring
    check (`"SUFFICIENT" in verdict`) is a bug waiting to happen (it also
    matches inside "INSUFFICIENT", which is why it needed a second check to
    exclude it). Structured output sidesteps that whole class of bug, and
    gets a confidence + one-line reason for free, useful for debugging why
    the agent decided to retry.
    """
    context = "\n\n".join(c["text"] for c in state["retrieved_chunks"])
    prompt = (
        "You are grading retrieved context for a question-answering system. "
        "Respond in JSON.\n\n"
        f"QUESTION: {state['question']}\n\n"
        f"RETRIEVED CONTEXT:\n{context[:6000]}\n\n"
        "Does this context contain enough information to answer the question? "
        "Reply with ONLY this JSON object, no other text:\n"
        '{"sufficient": <true or false>, "confidence": <0-100>, "reason": "<one short sentence>"}'
    )
    response = OPENAI_CLIENT.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=120,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        parsed = json.loads(response.choices[0].message.content)
        grade = "sufficient" if parsed.get("sufficient") else "insufficient"
    except (json.JSONDecodeError, TypeError):
        # Fail closed: if the judge call itself is broken, treat it as
        # insufficient and let the bounded rewrite loop try again, rather
        # than silently generating from context nobody actually vetted.
        grade, parsed = "insufficient", {}
    return {
        "grade": grade,
        "grade_confidence": parsed.get("confidence"),
        "grade_reason": parsed.get("reason"),
    }


def rewrite_node(state: RAGState) -> dict:
    """Ask the LLM to turn the question into a sharper retrieval query (also JSON mode)."""
    prompt = (
        "The following search query did not retrieve enough relevant context "
        "from a vector database of research papers. Respond in JSON.\n"
        f"{_history_block(state)}"
        f"ORIGINAL QUESTION: {state['question']}\n"
        f"SEARCH QUERY USED: {state.get('search_query') or state['question']}\n"
        f"WHY THE LAST ATTEMPT FELL SHORT: {state.get('grade_reason') or 'not enough matching content'}\n\n"
        "Rewrite it into a single, more effective search query — resolve any "
        "pronouns/references using the conversation history if relevant, add "
        "likely technical terms, expand acronyms, drop conversational filler. "
        'Reply with ONLY this JSON object, no other text: {"query": "<rewritten search query>"}'
    )
    response = OPENAI_CLIENT.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=100,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        new_query = (json.loads(response.choices[0].message.content).get("query") or "").strip()
    except (json.JSONDecodeError, TypeError):
        new_query = ""
    fallback = state.get("search_query") or state["question"]
    final_query = new_query or fallback
    # Collapse back to a single query on retry, even if the first pass was
    # decomposed — retrieve_node keys off sub_queries, so without this a
    # retry would silently keep re-running the *stale* sub-questions and
    # never actually use the freshly rewritten query.
    return {
        "search_query": final_query,
        "sub_queries": [final_query],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def route_after_retrieve(state: RAGState) -> str:
    """
    A genuinely decomposed (multi-document) retrieval has already had its
    correction applied by decompose_node. Grading it with the same
    single-answer-containment heuristic used for an ordinary question tends
    to call it "insufficient" even when it correctly pulled relevant
    material from *both* documents — a comparison's context rarely states
    the comparison verbatim, it just contains both halves of it — which
    would send it through rewrite_node, and rewrite_node collapses back to
    one query on retry (see its docstring), silently undoing the
    decomposition that was working. So a decomposed retrieval skips
    grading and goes straight to generation; only an un-decomposed
    (single-query) retrieval gets graded and is eligible for a retry.
    """
    if len(state.get("sub_queries") or []) > 1:
        return "end"
    return "grade"


def route_after_grade(state: RAGState) -> str:
    if state["grade"] == "sufficient" or state.get("retry_count", 0) >= MAX_RETRIES:
        return "end"
    return "rewrite"


# ---------------------------------------------------------------------------
# Graph assembly — this graph only decides what to answer with; see the
# module docstring for why generation happens outside it.
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("contextualize", contextualize_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)

    graph.set_entry_point("contextualize")
    graph.add_edge("contextualize", "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve, {"end": END, "grade": "grade"})
    graph.add_conditional_edges("grade", route_after_grade, {"end": END, "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")

    # MemorySaver = in-process checkpointer. Swap for a SQLite/Postgres
    # checkpointer if you want memory to survive a process restart.
    return graph.compile(checkpointer=MemorySaver())


AGENT = build_graph()


def prepare_turn(question: str, thread_id: str = "default") -> dict:
    """
    Run the retrieval / self-correction part of the agent for one question —
    contextualize, retrieve, grade, and rewrite-and-retry as needed — and
    return the resulting state (retrieved_chunks, grade info, retry_count,
    the conversation's chat_history *so far*). Does not call the LLM to
    generate an answer; pass the result to stream_answer() or run_turn() for
    that.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return AGENT.invoke({"question": question, "retry_count": 0}, config=config)


def stream_answer(question: str, state: dict, thread_id: str = "default"):
    """
    Stream the final answer for a question given the state prepare_turn()
    produced, yielding text as it's generated. Once the stream is exhausted,
    persists the completed answer (+ updated chat_history) back into the
    thread's checkpointed state, so the next prepare_turn() call for this
    thread_id sees it as prior conversation.
    """
    config = {"configurable": {"thread_id": thread_id}}
    parts = []
    for token in generate_answer_stream(question, state["retrieved_chunks"], state.get("chat_history")):
        parts.append(token)
        yield token

    answer = "".join(parts)
    AGENT.update_state(config, {
        "answer": answer,
        "chat_history": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    })


def run_turn(question: str, thread_id: str = "default") -> dict:
    """
    Non-streaming convenience wrapper: run prepare_turn() + stream_answer()
    back to back and return one dict with the complete answer included.
    For callers that don't need token-by-token output (eval.py, scripts).
    """
    state = prepare_turn(question, thread_id)
    state["answer"] = "".join(stream_answer(question, state, thread_id))
    return state
