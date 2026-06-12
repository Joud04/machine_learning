"""Verification manuelle de l'API /ask (l'API doit tourner sur localhost:8000).

Usage :
    python test_api.py
"""
import requests
import json

url = "http://localhost:8000/ask"
queries = [
    "Quelle est l'architecture de AssistKB v0 ?",
    "Qui a gagné la coupe du monde 1998 ?",
    "Comment configurer une collection Qdrant ?"
]


def main():
    for q in queries:
        print(f"\nQuery: {q}")
        resp = requests.post(url, json={"question": q})
        print(f"Status: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
