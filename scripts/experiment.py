"""Expérimentation 8.1 : impact du top-k sur la qualité du retrieval.

Fait varier k et compare recall@k, MRR et latence moyenne de recherche sur le
golden dataset, afin de justifier objectivement le choix de TOP_K en production.

Usage :
    python -m scripts.experiment
"""
import time

from app.config import TOP_K
from app.retrieve import RetrievalService
from eval.recall import evaluate, load_golden

K_VALUES = [1, 3, 5, 10]


def run() -> list[dict]:
    golden = load_golden()
    retriever = RetrievalService()
    rows = []
    for k in K_VALUES:
        start = time.perf_counter()
        report = evaluate(retriever, golden, k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rows.append({
            "k": k,
            "recall": report["recall"],
            "mrr": report["mrr"],
            "latency_ms": elapsed_ms / max(report["total"], 1),
        })
    return rows


def main() -> None:
    rows = run()

    print(f"\n=== Experimentation top-k (golden = {len(load_golden())} questions) ===")
    print(f"{'k':>3} | {'recall@k':>9} | {'MRR':>6} | {'latence moy (ms)':>16}")
    print("-" * 44)
    for r in rows:
        print(f"{r['k']:>3} | {r['recall']:>9.2f} | {r['mrr']:>6.3f} | {r['latency_ms']:>16.1f}")

    best = max(rows, key=lambda r: (round(r["recall"], 3), r["mrr"]))
    print(f"\nMeilleur compromis qualite : k={best['k']} "
          f"(recall={best['recall']:.2f}, MRR={best['mrr']:.3f}).")
    print(f"Valeur retenue en production : TOP_K={TOP_K} "
          f"(au-dela, le recall plafonne mais le contexte envoye au LLM grossit, "
          f"ce qui augmente cout et latence de generation).")


if __name__ == "__main__":
    main()
