# KnowledgeForge AI -- Cost & Latency Optimization Report

Benchmarked on 12 questions from the golden set.

## Latency (average per question)

| Pass | Avg latency (s) |
|---|---|
| Baseline (no optimization) | 2.05 |
| Optimized, cold cache | 3.63 |
| Optimized, warm cache | 0.08 |

**Latency improvement (baseline -> warm cache): 96%**

## Where requests were routed

- Cold cache pass: {'local': 9, 'cloud': 3}
- Warm cache pass: {'cache': 12}

## Estimated cost

- Baseline (all 12 questions to cloud): $0.0072
- Warm-cache pass (0 still hit cloud): $0.0000
- **Estimated savings: $0.0072 (100%)**

*Cost figures are estimates based on `EST_COST_PER_CLOUD_CALL_USD` in .env (free-tier APIs don't bill directly) -- they exist to make the savings concrete, not as an exact invoice. Local Ollama and cache hits are genuinely $0 regardless.*