import time
from unittest.mock import patch

import pytest

import src.llm as llm


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """These tests exercise real backoff logic -- don't actually wait 10-40s per test."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


class TestRetryBackoff:
    def test_succeeds_after_transient_rate_limit(self):
        calls = {"n": 0}

        def flaky(prompt, system=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("429 ResourceExhausted: quota exceeded")
            return "finally succeeded"

        with patch.object(llm, "_generate_gemini", side_effect=flaky):
            result = llm.generate("test", provider="gemini")

        assert result == "finally succeeded"
        assert calls["n"] == 3

    def test_non_rate_limit_errors_are_not_retried(self):
        calls = {"n": 0}

        def broken(prompt, system=None):
            calls["n"] += 1
            raise ValueError("some unrelated bug")

        with patch.object(llm, "_generate_gemini", side_effect=broken), pytest.raises(ValueError):
            llm.generate("test", provider="gemini")

        assert calls["n"] == 1, "should not waste retries on non-rate-limit errors"

    def test_gives_up_after_max_retries_on_persistent_rate_limit(self):
        calls = {"n": 0}

        class FakeUpstreamError(Exception):
            pass

        def always_limited(prompt, system=None):
            calls["n"] += 1
            raise FakeUpstreamError("429 quota exceeded")

        with patch.object(llm, "_generate_gemini", side_effect=always_limited), pytest.raises(FakeUpstreamError):
            llm.generate("test", provider="gemini")

        assert calls["n"] == 4  # initial attempt + 3 retries, never infinite

    def test_daily_quota_error_fails_immediately_no_wasted_wait(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

        daily_quota_msg = (
            '429 quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier" quota_value: 20'
        )

        def exhausted(prompt, system=None):
            raise Exception(daily_quota_msg)

        with patch.object(llm, "_generate_gemini", side_effect=exhausted), pytest.raises(RuntimeError) as exc_info:
            llm.generate("test", provider="gemini")

        assert sleep_calls == [], "a daily quota error must never trigger a backoff sleep"
        assert "Daily free-tier" in str(exc_info.value)
