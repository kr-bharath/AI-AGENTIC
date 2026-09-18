"""
Usage:
    python -m src.benchmark

Runs the golden set three ways and reports the difference:
  1. Baseline -- optimize=False: every question goes straight to the
     configured cloud provider, exactly like Phase 1-3.
  2. Optimized, cold cache -- optimize=True, cache cleared first: shows
     how many questions get deflected to the free local model immediately.
  3. Optimized, warm cache -- the SAME questions asked again: shows the
     cache doing its job (should mostly be instant "cache" hits).

Writes cost_optimization_report.md with latency and estimated $ saved.
"""
import time
from pathlib import Path
from statistics import mean

from . import cache, config
from .ask import ask
from .eval import GOLDEN_SET_PATH, load_golden_set

REPORT_PATH = Path("cost_optimization_report.md")


def _timed_pass(questions: list, optimize: bool) -> dict:
    routes = []
    latencies = []
    for q in questions:
        start = time.perf_counter()
        result = ask(q, optimize=optimize)
        latencies.append(time.perf_counter() - start)
        routes.append(result["route"])
    return {"routes": routes, "latencies": latencies}


def run_benchmark(golden_set_path: Path = GOLDEN_SET_PATH) -> dict:
    config.validate()
    golden_set = load_golden_set(golden_set_path)
    questions = [item["question"] for item in golden_set]

    cache.clear()

    print(f"Pass 1/3: baseline (no optimization), {len(questions)} questions...")
    baseline = _timed_pass(questions, optimize=False)

    cache.clear()  # baseline shouldn't have written to it anyway, but be certain

    print(f"Pass 2/3: optimized, cold cache, {len(questions)} questions...")
    cold = _timed_pass(questions, optimize=True)

    print("Pass 3/3: optimized, warm cache (same questions again)...")
    warm = _timed_pass(questions, optimize=True)

    def route_counts(routes):
        counts = {}
        for r in routes:
            key = r.split(" (")[0]  # collapse "cloud (local fallback)" -> "cloud"
            counts[key] = counts.get(key, 0) + 1
        return counts

    baseline_cost = len(questions) * config.EST_COST_PER_CLOUD_CALL_USD
    warm_cloud_calls = route_counts(warm["routes"]).get("cloud", 0)
    warm_cost = warm_cloud_calls * config.EST_COST_PER_CLOUD_CALL_USD

    return {
        "n_questions": len(questions),
        "baseline_avg_latency": mean(baseline["latencies"]),
        "cold_avg_latency": mean(cold["latencies"]),
        "warm_avg_latency": mean(warm["latencies"]),
        "cold_routes": route_counts(cold["routes"]),
        "warm_routes": route_counts(warm["routes"]),
        "baseline_cost_usd": baseline_cost,
        "warm_cost_usd": warm_cost,
    }


def write_report(summary: dict, path: Path = REPORT_PATH) -> None:
    saved_usd = summary["baseline_cost_usd"] - summary["warm_cost_usd"]
    saved_pct = (saved_usd / summary["baseline_cost_usd"] * 100) if summary["baseline_cost_usd"] else 0
    latency_improvement_pct = (
        (summary["baseline_avg_latency"] - summary["warm_avg_latency"])
        / summary["baseline_avg_latency"] * 100
        if summary["baseline_avg_latency"] else 0
    )

    lines = [
        "# KnowledgeForge AI -- Cost & Latency Optimization Report",
        "",
        f"Benchmarked on {summary['n_questions']} questions from the golden set.",
        "",
        "## Latency (average per question)",
        "",
        "| Pass | Avg latency (s) |",
        "|---|---|",
        f"| Baseline (no optimization) | {summary['baseline_avg_latency']:.2f} |",
        f"| Optimized, cold cache | {summary['cold_avg_latency']:.2f} |",
        f"| Optimized, warm cache | {summary['warm_avg_latency']:.2f} |",
        "",
        f"**Latency improvement (baseline -> warm cache): {latency_improvement_pct:.0f}%**",
        "",
        "## Where requests were routed",
        "",
        f"- Cold cache pass: {summary['cold_routes']}",
        f"- Warm cache pass: {summary['warm_routes']}",
        "",
        "## Estimated cost",
        "",
        f"- Baseline (all {summary['n_questions']} questions to cloud): ${summary['baseline_cost_usd']:.4f}",
        (
            f"- Warm-cache pass ({summary['warm_routes'].get('cloud', 0)} still hit cloud): "
            f"${summary['warm_cost_usd']:.4f}"
        ),
        f"- **Estimated savings: ${saved_usd:.4f} ({saved_pct:.0f}%)**",
        "",
        (
            "*Cost figures are estimates based on `EST_COST_PER_CLOUD_CALL_USD` in .env "
            "(free-tier APIs don't bill directly) -- they exist to make the savings "
            "concrete, not as an exact invoice. Local Ollama and cache hits are "
            "genuinely $0 regardless.*"
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    summary = run_benchmark()
    write_report(summary)

    print("\n--- Results ---")
    print(f"Baseline avg latency:      {summary['baseline_avg_latency']:.2f}s")
    print(f"Warm-cache avg latency:    {summary['warm_avg_latency']:.2f}s")
    print(f"Cold-cache routes:         {summary['cold_routes']}")
    print(f"Warm-cache routes:         {summary['warm_routes']}")
    print(f"Estimated baseline cost:   ${summary['baseline_cost_usd']:.4f}")
    print(f"Estimated warm-cache cost: ${summary['warm_cost_usd']:.4f}")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
