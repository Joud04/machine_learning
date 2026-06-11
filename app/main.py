import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import TOP_K, SIM_THRESHOLD
from app.metrics import log_request
from app.store import QdrantStore
from app.retrieve import RetrievalService
from app.generate import GenerationService, REFUSAL_TEXT


class QueryRequest(BaseModel):
    question: str
    k: int = TOP_K


class SourceItem(BaseModel):
    source: str
    position: int = 0
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    refused: bool
    best_score: float
    latency_ms: int
    tokens: dict


app = FastAPI(title="AssistKB Search API")

# Au premier demarrage le volume Qdrant est vierge : on cree la collection
# (vide) avant d'initialiser le wrapper LangChain, qui exige son existence.
QdrantStore().ensure_collection()

# Initialize Services
retriever = RetrievalService()
generator = GenerationService()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    RAG endpoint: retrieves context and generates a cited answer,
    or refuses when the best similarity score is below the threshold.
    """
    start_time = time.perf_counter()

    try:
        # 1. Retrieval phase (single vector search, scores included)
        results = retriever.retrieve_with_scores(request.question, k=request.k)
        best_score = results[0][1] if results else 0.0
        kept = [(doc, score) for doc, score in results if score >= SIM_THRESHOLD]

        # 2. Generation phase (skipped entirely on refusal: 0 token consumed)
        if not kept:
            answer, usage = REFUSAL_TEXT, {"input_tokens": 0, "output_tokens": 0}
            sources: List[SourceItem] = []
        else:
            answer, usage = generator.generate(request.question, [doc for doc, _ in kept])
            sources = [
                SourceItem(
                    source=doc.metadata.get("source", "inconnue"),
                    position=doc.metadata.get("position", 0),
                    score=round(score, 3),
                )
                for doc, score in kept
            ]

        # Deuxieme ligne de defense : le seuil peut laisser passer une question
        # hors-corpus, mais le prompt impose alors au LLM le texte de refus exact.
        # On le detecte pour rester coherent (refused=true, pas de sources citees).
        refused = not kept or REFUSAL_TEXT in answer
        if refused:
            sources = []

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request({
            "question": request.question,
            "refused": refused,
            "best_score": round(best_score, 3),
            "latency_ms": latency_ms,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "top_k": request.k,
            "threshold": SIM_THRESHOLD,
        })
        return QueryResponse(
            answer=answer,
            sources=sources,
            refused=refused,
            best_score=round(best_score, 3),
            latency_ms=latency_ms,
            tokens=usage,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
