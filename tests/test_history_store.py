"""
Tests for conversation history storage (local backend).

This is the piece that makes the service stateless — instances are only
fungible if history reliably round-trips through an external store — so the
contract is worth pinning down even though the local backend is a dict.
"""

import history_store


class TestLocalHistoryStore:
    def setup_method(self):
        self.thread = "test-thread"
        history_store.reset(self.thread)

    def test_unknown_thread_has_empty_history(self):
        assert history_store.get_history("never-seen") == []

    def test_empty_thread_id_is_not_stored(self):
        assert history_store.get_history("") == []
        assert history_store.append_turn("", "q", "a") == []

    def test_append_round_trips_a_turn(self):
        history_store.append_turn(self.thread, "What is X?", "X is a thing.")
        history = history_store.get_history(self.thread)

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "What is X?"}
        assert history[1] == {"role": "assistant", "content": "X is a thing."}

    def test_turns_accumulate_in_order(self):
        history_store.append_turn(self.thread, "first?", "one")
        history_store.append_turn(self.thread, "second?", "two")

        contents = [h["content"] for h in history_store.get_history(self.thread)]
        assert contents == ["first?", "one", "second?", "two"]

    def test_threads_are_isolated(self):
        history_store.append_turn("thread-a", "a?", "a!")
        history_store.append_turn("thread-b", "b?", "b!")
        try:
            assert len(history_store.get_history("thread-a")) == 2
            assert history_store.get_history("thread-a")[0]["content"] == "a?"
            assert history_store.get_history("thread-b")[0]["content"] == "b?"
        finally:
            history_store.reset("thread-a")
            history_store.reset("thread-b")

    def test_history_is_capped(self):
        """
        Unbounded history would eventually blow DynamoDB's 400 KB item limit
        and quietly inflate every prompt.
        """
        for i in range(history_store.MAX_TURNS + 10):
            history_store.append_turn(self.thread, f"q{i}", f"a{i}")

        history = history_store.get_history(self.thread)
        assert len(history) == history_store.MAX_TURNS * 2
        # The most recent turn survives; the oldest is evicted.
        assert history[-1]["content"] == f"a{history_store.MAX_TURNS + 9}"
        assert "q0" not in [h["content"] for h in history]

    def test_get_history_returns_a_copy(self):
        """Callers mutating the returned list must not corrupt the store."""
        history_store.append_turn(self.thread, "q", "a")
        history_store.get_history(self.thread).append({"role": "user", "content": "injected"})
        assert len(history_store.get_history(self.thread)) == 2

    def test_reset_clears_a_thread(self):
        history_store.append_turn(self.thread, "q", "a")
        history_store.reset(self.thread)
        assert history_store.get_history(self.thread) == []
