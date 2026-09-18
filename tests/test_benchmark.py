import json
import time
from unittest.mock import patch

import src.benchmark as bench

from .conftest import make_vec, requires_redis

FAKE_CHUNKS = [{"source": "doc.md", "chunk_index": 0, "content": "some content", "distance": 0.05}]


@requires_redis
class TestBenchmark:
    def test_cold_pass_routes_locally_and_warm_pass_hits_cache(self, tmp_path):
        tiny = [{"id": "b1", "question": "What is RAG?"}, {"id": "b2", "question": "What is chunking?"}]
        golden_path = tmp_path / "tiny_bench.json"
        golden_path.write_text(json.dumps(tiny))

        vecs = {"What is RAG?": make_vec(1), "What is chunking?": make_vec(2)}

        def fake_embed(q):
            return vecs[q]

        def fake_generate(prompt, system=None, provider=None):
            time.sleep(0.01 if provider == "local" else 0.03)
            return f"Answer via {provider or 'default'}"

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", side_effect=fake_embed), \
             patch("src.cache.embed_query", side_effect=fake_embed), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            summary = bench.run_benchmark(golden_set_path=golden_path)

        # both questions are short, no complexity keywords -> route local on cold pass
        assert summary["cold_routes"] == {"local": 2}
        # same questions again -> should be all cache hits
        assert summary["warm_routes"] == {"cache": 2}
        # nothing hit cloud on the warm pass -> zero cost
        assert summary["warm_cost_usd"] == 0.0
        # warm cache must be faster than the uncached baseline
        assert summary["warm_avg_latency"] < summary["baseline_avg_latency"]

    def test_report_renders_with_correct_numbers(self, tmp_path):
        summary = {
            "n_questions": 2,
            "baseline_avg_latency": 2.0,
            "cold_avg_latency": 0.5,
            "warm_avg_latency": 0.05,
            "cold_routes": {"local": 2},
            "warm_routes": {"cache": 2},
            "baseline_cost_usd": 0.0012,
            "warm_cost_usd": 0.0,
        }
        report_path = tmp_path / "report.md"
        bench.write_report(summary, report_path)
        content = report_path.read_text()
        assert "100%" in content  # 100% cost savings when warm_cost_usd is 0
        assert "0.05" in content
