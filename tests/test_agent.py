from unittest.mock import patch

import src.agent
import src.db

from .conftest import make_vec

FAKE_CHUNKS = [
    {"source": "doc_a.md", "chunk_index": 0, "content": "Doc A discusses chunking strategy.", "distance": 0.1},
    {"source": "doc_a.md", "chunk_index": 1, "content": "Doc A recommends 500-word chunks.", "distance": 0.12},
    {"source": "doc_b.md", "chunk_index": 0, "content": "Doc B discusses embedding models.", "distance": 0.15},
]


def _fake_generate_factory(fail_first_check: bool):
    call_log = []

    def fake_generate(prompt, system=None, provider=None):
        call_log.append({"system": (system or "")[:40]})
        sys_lower = (system or "").lower()
        if "classify" in sys_lower:
            return "MULTI" if "compare" in prompt.lower() else "SIMPLE"
        if "strict reviewer" in sys_lower:
            if not fail_first_check:
                return "PASS"
            fails_so_far = sum(1 for c in call_log if "strict reviewer" in c["system"])
            return "PASS" if fails_so_far > 1 else "FAIL: table missing"
        if "research analyst" in sys_lower:
            return "| Doc | Point |\n|---|---|\n| A | chunking |\n| B | embeddings |"
        return "Simple grounded answer."

    return fake_generate, call_log


class TestAgentRouting:
    def test_simple_question_skips_self_check(self):
        fake_generate, _ = _fake_generate_factory(fail_first_check=False)
        with patch("src.agent.generate", side_effect=fake_generate), \
             patch("src.agent.embed_query", return_value=make_vec(1)), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            result = src.agent.run_agent("What chunk size does doc A recommend?")

        assert result["task_type"] == "simple"
        assert result["answer"] == "Simple grounded answer."
        assert result["check_passed"] is True

    def test_multi_step_retries_once_on_failed_check_then_passes(self):
        fake_generate, _ = _fake_generate_factory(fail_first_check=True)
        with patch("src.agent.generate", side_effect=fake_generate), \
             patch("src.agent.embed_query", return_value=make_vec(2)), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            result = src.agent.run_agent("Compare doc A and doc B in a table")

        assert result["task_type"] == "multi_step"
        assert result["retry_count"] == 2  # one fail + one pass = 2 self-check calls
        assert result["check_passed"] is True

    def test_no_documents_short_circuits_cleanly(self):
        fake_generate, _ = _fake_generate_factory(fail_first_check=False)
        with patch("src.agent.generate", side_effect=fake_generate), \
             patch("src.agent.embed_query", return_value=make_vec(3)), \
             patch("src.db.search", return_value=[]):
            result = src.agent.run_agent("Anything")

        assert "No documents" in result["answer"]

    def test_retry_loop_terminates_even_if_check_never_passes(self):
        call_count = {"self_check": 0}

        def always_fail(prompt, system=None, provider=None):
            sys_lower = (system or "").lower()
            if "classify" in sys_lower:
                return "MULTI"
            if "strict reviewer" in sys_lower:
                call_count["self_check"] += 1
                return "FAIL: never good enough"
            return "some comparison answer"

        with patch("src.agent.generate", side_effect=always_fail), \
             patch("src.agent.embed_query", return_value=make_vec(4)), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            result = src.agent.run_agent("Compare things in a table")

        assert call_count["self_check"] == 2, "must stop after exactly one retry, never loop forever"
        assert result["check_passed"] is False
