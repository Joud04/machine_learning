"""Metriques qualite + exploitation du RAG.

L'API journalise chaque requete dans data/requests.jsonl (log_request).
Ce module agrege ensuite les logs en indicateurs chiffres :

    python -m app.metrics
"""
import json
import pathlib
import statistics

LOG_PATH = pathlib.Path("data/requests.jsonl")

# Tarifs Groq llama-3.1-8b-instant (USD / 1M tokens)
PRICE_IN = 0.05
PRICE_OUT = 0.08


def log_request(entry: dict) -> None:
    """Ajoute une ligne JSONL (appele par l'API a chaque requete)."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def percentile(values: list[float], pct: float) -> float:
    """Percentile par interpolation basse (suffisant pour nos volumes)."""
    ordered = sorted(values)
    index = max(0, int(round(pct / 100 * len(ordered) + 0.5)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def report() -> dict:
    if not LOG_PATH.exists():
        raise SystemExit("Aucune requete loggee : appeler POST /ask d'abord.")
    entries = [json.loads(line) for line in LOG_PATH.open(encoding="utf-8")]
    if not entries:
        raise SystemExit("Journal vide : appeler POST /ask d'abord.")

    latencies = [e["latency_ms"] for e in entries]
    answered = [e for e in entries if not e["refused"]]
    avg_in = statistics.mean(e["input_tokens"] for e in answered) if answered else 0
    avg_out = statistics.mean(e["output_tokens"] for e in answered) if answered else 0
    cost_per_1000 = (avg_in * PRICE_IN + avg_out * PRICE_OUT) / 1_000_000 * 1000

    return {
        "nb_requetes": len(entries),
        "score_similarite_moyen_top1": round(statistics.mean(e["best_score"] for e in entries), 3),
        "taux_refus_pct": round(100 * sum(e["refused"] for e in entries) / len(entries), 1),
        "latence_p50_ms": percentile(latencies, 50),
        "latence_p95_ms": percentile(latencies, 95),
        "tokens_entree_moyens": round(avg_in),
        "tokens_sortie_moyens": round(avg_out),
        "cout_projete_usd_pour_1000_questions": round(cost_per_1000, 4),
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, ensure_ascii=False))
