"""
observability.py
Structured logging and metrics.

Deliberately *not* OpenTelemetry. Getting an ADOT sidecar wired up correctly
is a real yak-shave, and for a single service the useful signal is: one
structured line per request that you can query in CloudWatch Logs Insights,
plus a handful of numbers you can graph and alarm on.

CloudWatch EMF (Embedded Metric Format) is how you get the second one for
free: emit a JSON log line shaped a particular way and CloudWatch parses it
into real custom metrics — no agent, no sidecar, no extra infrastructure.
Locally it's just a JSON line on stdout.
"""

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar

SERVICE_NAME = os.getenv("SERVICE_NAME", "rag-api")
METRIC_NAMESPACE = os.getenv("METRIC_NAMESPACE", "RagApi")
BUILD_SHA = os.getenv("BUILD_SHA", "dev")
EMIT_EMF = os.getenv("EMIT_EMF", "true").lower() == "true"

# Set per request by the middleware so every log line in a request can be
# correlated without threading an argument through every function.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """One JSON object per line — the format CloudWatch Logs Insights can query."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "build": BUILD_SHA,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={"fields": {...}})
        if hasattr(record, "fields") and isinstance(record.fields, dict):
            payload.update(record.fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    # uvicorn's own access log duplicates what our middleware records, in a
    # different (non-JSON) shape.
    logging.getLogger("uvicorn.access").disabled = True
    return logging.getLogger(SERVICE_NAME)


log = configure_logging()


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def emit_metrics(metrics: dict[str, float], dimensions: dict[str, str] | None = None) -> None:
    """
    Emit an EMF log line so CloudWatch turns these into real custom metrics.

    metrics: name -> value (numbers only)
    dimensions: low-cardinality labels only. Never put request_id or
    thread_id here — every distinct dimension value bills as its own metric.
    """
    if not metrics:
        return
    if not EMIT_EMF:
        log.info("metrics", extra={"fields": {"metrics": metrics}})
        return

    dims = {"Service": SERVICE_NAME, **(dimensions or {})}
    payload = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": METRIC_NAMESPACE,
                "Dimensions": [list(dims.keys())],
                "Metrics": [{"Name": name} for name in metrics],
            }],
        },
        **dims,
        **metrics,
        "request_id": request_id_var.get(),
    }
    print(json.dumps(payload, default=str), flush=True)


# Rough per-1K-token pricing for gpt-4o-mini, so cost shows up on the same
# dashboard as latency. Update if the model or pricing changes.
_COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_INPUT", "0.00015"))
_COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_OUTPUT", "0.0006"))


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens / 1000) * _COST_PER_1K_INPUT
        + (completion_tokens / 1000) * _COST_PER_1K_OUTPUT,
        6,
    )


class Timer:
    """with Timer() as t: ...  ->  t.ms"""

    def __enter__(self):
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self._start) * 1000, 1)
        return False
