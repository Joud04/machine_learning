"""Extraction du corpus (HTML/PDF/JSON/TXT/CSV) puis chunking -> data/chunks.jsonl."""
import json
import pathlib
import uuid

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selectolax.parser import HTMLParser

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

CORPUS_DIRS = [pathlib.Path("corpus/seed"), pathlib.Path("corpus/raw")]
OUT_PATH = pathlib.Path("data/chunks.jsonl")
MIN_TEXT_LEN = 100  # en dessous, document vide ou inexploitable


def extract_text(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        tree = HTMLParser(path.read_text(encoding="utf-8", errors="ignore"))
        for tag in tree.css("script, style"):
            tag.decompose()
        return tree.body.text(separator="\n") if tree.body else ""
    if suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, ensure_ascii=False, indent=1)
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def build_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for corpus_dir in CORPUS_DIRS:
        if not corpus_dir.exists():
            continue
        for path in sorted(corpus_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                text = extract_text(path).strip()
            except Exception as exc:
                print(f"[erreur] {path}: {exc}")
                continue
            if len(text) < MIN_TEXT_LEN:
                print(f"[skip] {path} (vide ou trop court)")
                continue
            for position, piece in enumerate(splitter.split_text(text)):
                chunks.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.as_posix()}:{position}")),
                    "text": piece,
                    "metadata": {
                        "source": path.as_posix(),
                        "doc_type": path.suffix.lstrip("."),
                        "position": position,
                    },
                })
            print(f"[ok] {path}")
    return chunks


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(exist_ok=True)
    all_chunks = build_chunks()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"{len(all_chunks)} chunks ecrits dans {OUT_PATH}")
