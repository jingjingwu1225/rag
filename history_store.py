"""
history_store.py
Conversation history, stored outside the process.

Why this exists: the agent originally kept chat history in LangGraph's
in-process MemorySaver checkpointer. That works for one CLI or one Streamlit
session, but it makes the service *stateful* — turn 2 of a conversation that
lands on a different instance (or a different uvicorn worker) finds no
history, so contextualize_node skips resolving the follow-up, retrieval runs
on the raw pronoun, and the user gets a confidently wrong answer with no
error anywhere. It also grows without bound: no TTL, no eviction.

Externalizing it to DynamoDB makes instances fungible — any instance can
serve any turn of any conversation — which is what lets the service scale
horizontally at all. TTL handles cleanup.

Two backends:
  local    — in-process dict. For tests, CLIs, and running without AWS.
  dynamodb — one table, PK `thread_id`, `history` (JSON), `ttl` (epoch secs).

Chosen via HISTORY_BACKEND. Defaults to local so nothing breaks offline.
"""

import json
import os
import threading
import time

HISTORY_BACKEND = os.getenv("HISTORY_BACKEND", "local")
HISTORY_TABLE = os.getenv("HISTORY_TABLE", "rag-api-history")
HISTORY_TTL_SECONDS = int(os.getenv("HISTORY_TTL_SECONDS", str(24 * 60 * 60)))
# Cap stored history so a long-running thread can't grow the item past
# DynamoDB's 400 KB limit (and can't quietly inflate every prompt).
MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "20"))


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------
_LOCAL: dict[str, list] = {}
_LOCAL_LOCK = threading.Lock()


def _local_get(thread_id: str) -> list:
    with _LOCAL_LOCK:
        return list(_LOCAL.get(thread_id, []))


def _local_put(thread_id: str, history: list) -> None:
    with _LOCAL_LOCK:
        _LOCAL[thread_id] = history


# ---------------------------------------------------------------------------
# DynamoDB backend
# ---------------------------------------------------------------------------
_TABLE = None
_TABLE_LOCK = threading.Lock()


def _table():
    global _TABLE
    if _TABLE is None:
        with _TABLE_LOCK:
            if _TABLE is None:
                import boto3  # imported lazily so local/test use needs no AWS deps

                _TABLE = boto3.resource("dynamodb").Table(HISTORY_TABLE)
    return _TABLE


def _ddb_get(thread_id: str) -> list:
    response = _table().get_item(Key={"thread_id": thread_id})
    item = response.get("Item")
    if not item or "history" not in item:
        return []
    try:
        return json.loads(item["history"])
    except (json.JSONDecodeError, TypeError):
        # A corrupt item shouldn't take down the conversation — treat it as
        # a fresh thread rather than 500ing the request.
        return []


def _ddb_put(thread_id: str, history: list) -> None:
    _table().put_item(Item={
        "thread_id": thread_id,
        "history": json.dumps(history),
        "ttl": int(time.time()) + HISTORY_TTL_SECONDS,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_history(thread_id: str) -> list[dict]:
    """Return the stored [{'role','content'}, ...] for a thread, oldest first."""
    if not thread_id:
        return []
    if HISTORY_BACKEND == "dynamodb":
        return _ddb_get(thread_id)
    return _local_get(thread_id)


def append_turn(thread_id: str, question: str, answer: str) -> list[dict]:
    """
    Append one [user, assistant] exchange and persist it. Returns the new
    history. Read-modify-write is fine here: a single conversation thread is
    inherently sequential (one user typing), so there's no realistic
    concurrent-writer case to lose.
    """
    if not thread_id:
        return []
    history = get_history(thread_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    history = history[-(MAX_TURNS * 2):]

    if HISTORY_BACKEND == "dynamodb":
        _ddb_put(thread_id, history)
    else:
        _local_put(thread_id, history)
    return history


def reset(thread_id: str) -> None:
    """Drop a thread's history (used by tests and the UI's 'new conversation')."""
    if HISTORY_BACKEND == "dynamodb":
        _table().delete_item(Key={"thread_id": thread_id})
    else:
        with _LOCAL_LOCK:
            _LOCAL.pop(thread_id, None)
