"""
Shared test configuration.

The whole suite is designed to run with no network and no credentials — the
model is mocked and the agent is stubbed. This file *enforces* that rather
than leaving it to convention.

Why it matters: a developer's local .env is loaded automatically by
python-dotenv at import, so anything that quietly depends on a real key
passes locally and fails in CI, where no .env exists. That divergence bit
this project twice — an import-time config check, then a startup validation
call — both of which looked fine locally and broke the CI run. Clearing the
key for the whole session makes local runs behave exactly like CI.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def no_ambient_credentials():
    """Remove any real API key for the duration of the test session."""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    yield
    if saved is not None:
        os.environ["OPENAI_API_KEY"] = saved
