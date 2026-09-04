"""
eval.py
A small, reference-free evaluation harness for the RAG pipeline — the kind
of thing tools like RAGAS automate, built here from scratch so it's obvious
what's actually being measured.

For each question in eval_questions.json, it runs the full agentic pipeline
(agent_graph.run_turn) and then uses an LLM judge to score two axes that
don't require a hand-written "gold answer":

  faithfulness  (1-5) : is every claim in the answer actually supported by
                        the retrieved context, or did the model make
                        something up? This is the #1 thing to catch in RAG.
  relevancy     (1-5) : does the answer actually address the question asked?

It also reports how many questions needed a query rewrite (retry_count > 0),
as a proxy for how often naive single-shot retrieval would have failed.

Run:
    python eval.py
"""

import json
import os
import re
import statistics
import sys

from agent_graph import run_turn
from rag_core import GENERATION_MODEL, openai_client

_HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.getenv("EVAL_QUESTIONS_FILE", os.path.join(_HERE, "eval_questions.json"))
RESULTS_FILE = os.getenv("EVAL_RESULTS_FILE", os.path.join(_HERE, "eval_results.json"))

# CI gate thresholds. Conservative on purpose: gated on the MEAN over the
# question set, not per-question, so one judge wobble doesn't block a deploy.
MIN_FAITHFULNESS = float(os.getenv("MIN_FAITHFULNESS", "4.0"))
MIN_RELEVANCY = float(os.getenv("MIN_RELEVANCY", "4.0"))

JUDGE_PROMPT = """You are evaluating one answer from a RAG (retrieval-augmented \
generation) system. Score it on two axes, each 1-5 (5 = best).

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

SYSTEM'S ANSWER:
{answer}

faithfulness (1-5): Is every claim in the answer supported by the retrieved \
context? 5 = fully grounded, no fabrication (this includes correctly saying \
"I don't know" when the context doesn't cover it). 1 = confidently made \
things up that aren't in the context.

relevancy (1-5): Does the answer actually address what was asked? 5 = fully \
on-topic and useful. 1 = off-topic or non-responsive.

Reply with ONLY this JSON, no other text:
{{"faithfulness": <int>, "relevancy": <int>, "reason": "<one sentence>"}}
"""


def judge(question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context[:4000], answer=answer)
    response = openai_client().chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=150,
        temperature=0,  # a judge that scores differently run-to-run makes the
                        # CI gate flaky — we saw exactly that (5.00 then 4.33
                        # on identical answers) before pinning this
        messages=[{"role": "user", "content": prompt}],
    )
    raw = (response.choices[0].message.content or "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {"faithfulness": None, "relevancy": None, "reason": raw}
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevancy": None, "reason": f"(unparsable judge output: {raw})"}


def main():
    with open(QUESTIONS_FILE) as f:
        eval_questions = json.load(f)

    results = []
    for item in eval_questions:
        print(f"Evaluating [{item['id']}]: {item['question']}")
        # No chat_history: every eval question is judged standalone, so one
        # question's answer can't leak into the next one's context.
        result = run_turn(item["question"])
        context = "\n\n".join(c["text"] for c in result["retrieved_chunks"])
        scores = judge(item["question"], context, result["answer"])

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": result["answer"],
            "sources": [c["source"] for c in result["retrieved_chunks"]],
            "retries": result.get("retry_count", 0),
            **scores,
        })
        print(f"  faithfulness={scores.get('faithfulness')} relevancy={scores.get('relevancy')} "
              f"retries={result.get('retry_count', 0)} -> {scores.get('reason')}")

    faith_scores = [r["faithfulness"] for r in results if isinstance(r["faithfulness"], int)]
    rel_scores = [r["relevancy"] for r in results if isinstance(r["relevancy"], int)]
    retried = sum(1 for r in results if r["retries"] > 0)

    mean_faith = statistics.mean(faith_scores) if faith_scores else 0.0
    mean_rel = statistics.mean(rel_scores) if rel_scores else 0.0

    print("\n" + "=" * 60)
    print(f"Questions evaluated:       {len(results)}")
    print(f"Avg faithfulness (1-5):    {mean_faith:.2f}  (gate: >= {MIN_FAITHFULNESS})")
    print(f"Avg relevancy (1-5):       {mean_rel:.2f}  (gate: >= {MIN_RELEVANCY})")
    print(f"Needed a query rewrite:    {retried}/{len(results)}")
    print("=" * 60)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {RESULTS_FILE}")

    # Exit non-zero on breach so this can gate a deploy in CI. Without this
    # the workflow would happily ship a regression that the eval just measured.
    failures = []
    if not faith_scores:
        failures.append("no faithfulness scores parsed from the judge")
    elif mean_faith < MIN_FAITHFULNESS:
        failures.append(f"faithfulness {mean_faith:.2f} < {MIN_FAITHFULNESS}")
    if rel_scores and mean_rel < MIN_RELEVANCY:
        failures.append(f"relevancy {mean_rel:.2f} < {MIN_RELEVANCY}")

    if failures:
        print("\nEVAL GATE FAILED: " + "; ".join(failures))
        sys.exit(1)
    print("\nEval gate passed.")


if __name__ == "__main__":
    main()
