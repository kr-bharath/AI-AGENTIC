import json
from unittest.mock import patch

import src.eval as ev


def _write_tiny_golden_set(tmp_path):
    tiny = [{"id": "t1", "question": "What is RAG?"}, {"id": "t2", "question": "Why use RAG?"}]
    p = tmp_path / "tiny_golden.json"
    p.write_text(json.dumps(tiny))
    return p


FAKE_CHUNKS = [
    {"source": "rag_basics.md", "chunk_index": 0, "content": "RAG combines retrieval and generation.", "distance": 0.05}
]


class TestEvalHarness:
    def test_end_to_end_with_well_formed_judge_output(self, tmp_path):
        golden_path = _write_tiny_golden_set(tmp_path)

        def fake_generate(prompt, system=None, provider=None):
            if "impartial evaluator" in (system or "").lower():
                return '{"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8, "notes": "Good."}'
            return "RAG combines retrieval and generation to ground answers."

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.eval.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", return_value=[0.0] * 384), \
             patch("src.eval.embed_query", return_value=[0.0] * 384), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            summary = ev.run_eval(golden_set_path=golden_path)

        assert summary["avg_faithfulness"] == 0.9
        assert len(summary["results"]) == 2

    def test_judge_output_wrapped_in_markdown_fences_is_parsed(self, tmp_path):
        golden_path = _write_tiny_golden_set(tmp_path)

        def fake_generate(prompt, system=None, provider=None):
            if "impartial evaluator" in (system or "").lower():
                return '```json\n{"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "notes": "ok"}\n```'
            return "some answer"

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.eval.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", return_value=[0.0] * 384), \
             patch("src.eval.embed_query", return_value=[0.0] * 384), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            summary = ev.run_eval(golden_set_path=golden_path)

        assert summary["avg_faithfulness"] == 0.5

    def test_non_json_judge_output_degrades_to_scored_failure_not_a_crash(self, tmp_path):
        golden_path = _write_tiny_golden_set(tmp_path)

        def fake_generate(prompt, system=None, provider=None):
            if "impartial evaluator" in (system or "").lower():
                return "I think this is pretty good overall!"
            return "some answer"

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.eval.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", return_value=[0.0] * 384), \
             patch("src.eval.embed_query", return_value=[0.0] * 384), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            summary = ev.run_eval(golden_set_path=golden_path)

        assert summary["avg_faithfulness"] == 0.0
        assert "PARSE ERROR" in summary["results"][0]["notes"]

    def test_report_renders_correctly(self, tmp_path):
        fake_summary = {
            "results": [
                {"id": "q01", "question": "What is RAG?", "answer": "...", "faithfulness": 0.9,
                 "answer_relevancy": 0.85, "context_precision": 0.8, "notes": "Good."},
            ],
            "avg_faithfulness": 0.9,
            "avg_answer_relevancy": 0.85,
            "avg_context_precision": 0.8,
        }
        report_path = tmp_path / "report.md"
        ev.write_report(fake_summary, report_path)
        content = report_path.read_text()
        assert "0.90" in content
        assert "q01" in content
