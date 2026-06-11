"""Évaluation de la qualité du retrieval sur le golden dataset.

Mesure deux indicateurs standards, sans appel LLM (retrieval seul) :
  - recall@k : proportion de questions dont la source attendue figure dans le top-k.
  - MRR (Mean Reciprocal Rank) : moyenne de 1/rang de la bonne source (sensible
    à la position, pas seulement à la présence).

Usage :
    python -m eval.recall          # k = TOP_K (config)
    python -m eval.recall 3        # k impose
"""
import json
import pathlib
import sys

from app.config import TOP_K
from app.retrieve import RetrievalService

GOLDEN_PATH = pathlib.Path("eval/golden.jsonl")


def load_golden(path: pathlib.Path = GOLDEN_PATH) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def rank_of_expected(retriever: RetrievalService, question: str,
                     expected_source: str, k: int) -> int | None:
    """Rang (1-based) de la source attendue dans le top-k, ou None si absente."""
    results = retriever.retrieve_with_scores(question, k=k)
    for rank, (doc, _score) in enumerate(results, start=1):
        if doc.metadata.get("source") == expected_source:
            return rank
    return None


def evaluate(retriever: RetrievalService, golden: list[dict], k: int) -> dict:
    """Calcule recall@k et MRR sur l'ensemble du golden dataset."""
    hits = 0
    reciprocal_ranks = []
    details = []
    for item in golden:
        rank = rank_of_expected(retriever, item["question"], item["expected_source"], k)
        hits += int(rank is not None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        details.append({
            "question": item["question"],
            "expected_source": item["expected_source"],
            "rank": rank,
        })
    n = len(golden) or 1
    return {
        "k": k,
        "recall": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "hits": hits,
        "total": len(golden),
        "details": details,
    }


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else TOP_K
    golden = load_golden()
    if not golden:
        print("Golden dataset vide.")
        return

    retriever = RetrievalService()
    report = evaluate(retriever, golden, k)

    print(f"\n=== Evaluation retrieval (k={k}, {report['total']} questions) ===")
    for d in report["details"]:
        status = f"rang {d['rank']}" if d["rank"] else "MANQUE"
        flag = "OK " if d["rank"] else "KO "
        print(f"  [{flag}] {status:<8} | {d['question'][:60]}")
    print(f"\nrecall@{k} = {report['recall']:.2f}  ({report['hits']}/{report['total']})")
    print(f"MRR       = {report['mrr']:.3f}")


if __name__ == "__main__":
    main()
