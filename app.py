"""
app.py
Chat UI for the agentic, memory-aware RAG system — wraps agent_graph.py
(contextualize -> retrieve+rerank -> grade -> rewrite loop -> stream answer)
instead of a single fixed rag_core pass, so the demo people click through
actually shows the same self-correction and follow-up memory as
agent_query.py, not just the plain retrieval path.

Run `python ingest.py` first, then:

    streamlit run app.py
"""

import uuid

import streamlit as st

import history_store
from agent_graph import prepare_turn, stream_answer
from rag_core import summarize_sources

st.set_page_config(page_title="My Research RAG", page_icon="📄")
st.title("📄 Ask My Papers")
st.caption(
    "Agentic Q&A over my own published research + methodology docs — "
    "self-correcting retrieval, reranked sources, and follow-up memory "
    "within this session."
)

# One LangGraph thread per browser session, so follow-up questions resolve
# ("what about that?") the same way they do in agent_query.py. Streamlit
# reruns this whole script on every interaction, so both the thread_id and
# the on-screen transcript have to live in session_state to survive that.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}], display only

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about the research...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, and grading context..."):
            state = prepare_turn(question, history_store.get_history(st.session_state.thread_id))

        if not state.get("retrieved_chunks"):
            st.warning("No relevant chunks found — did you run `python ingest.py` yet?")
        else:
            if state.get("retry_count", 0) > 0:
                st.caption(
                    f"↻ Initial retrieval was graded insufficient — query rewritten "
                    f"{state['retry_count']}x → \"{state['search_query']}\""
                )

            thread_id = st.session_state.thread_id
            answer = st.write_stream(stream_answer(
                question,
                state,
                on_complete=lambda a: history_store.append_turn(thread_id, question, a),
            ))
            st.session_state.messages.append({"role": "assistant", "content": answer})

            chunks = state["retrieved_chunks"]
            sources = summarize_sources(chunks)
            # Group by source and show a count per file, not one repeated
            # filename per chunk — several distinct passages from the same
            # paper (common for a question fully answerable from one) would
            # otherwise read as the same source "duplicated" in the list.
            label = (
                f"Sources retrieved — {len(chunks)} passage{'s' if len(chunks) != 1 else ''} "
                f"from {len(sources)} document{'s' if len(sources) != 1 else ''}"
            )
            with st.expander(label):
                grouped: dict[str, list] = {}
                for c in chunks:
                    grouped.setdefault(c["source"], []).append(c)
                for src, src_chunks in grouped.items():
                    st.markdown(f"**{src}** — {len(src_chunks)} passage{'s' if len(src_chunks) != 1 else ''}")
                    for i, c in enumerate(src_chunks, 1):
                        # distance/bm25_score can each be None — a hybrid
                        # hit found by only one of the two methods (e.g. an
                        # exact-keyword match vector search ranked outside
                        # the top-k) never gets the other method's field.
                        dist = f"{c['distance']:.3f}" if c.get("distance") is not None else "n/a"
                        bm25 = f"{c['bm25_score']:.2f}" if c.get("bm25_score") is not None else "n/a"
                        st.caption(
                            f"Passage {i} · rerank score {c.get('rerank_score')} · "
                            f"distance {dist} · bm25 {bm25}"
                        )
                        st.write(c["text"])
                        st.divider()

if st.session_state.messages and st.sidebar.button("Start a new conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()
