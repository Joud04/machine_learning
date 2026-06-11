"""Recupere des documents publics depuis data.gouv.fr selon une requete."""
import sys
import pathlib

import requests

API = "https://www.data.gouv.fr/api/1/datasets/"
ALLOWED = {"pdf", "json", "csv", "txt"}
MAX_BYTES = 5_000_000  # on ignore les ressources trop volumineuses


def fetch(query: str, n_docs: int, out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resp = requests.get(API, params={"q": query, "page_size": 20}, timeout=30)
    resp.raise_for_status()
    downloaded = 0
    for dataset in resp.json().get("data", []):
        for res in dataset.get("resources", []):
            if downloaded >= n_docs:
                break
            fmt = (res.get("format") or "").lower()
            url = res.get("url")
            if fmt not in ALLOWED or not url:
                continue
            name = f"{dataset['slug'][:40]}_{downloaded}.{fmt}"
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                if len(r.content) > MAX_BYTES:
                    print(f"[skip] {name} (trop volumineux)")
                    continue
                (out / name).write_bytes(r.content)
                downloaded += 1
                print(f"[ok] {name} ({len(r.content)} octets)")
            except requests.RequestException as exc:
                print(f"[skip] {url}: {exc}")
        if downloaded >= n_docs:
            break
    print(f"{downloaded} documents telecharges dans {out}")


if __name__ == "__main__":
    fetch(sys.argv[1], int(sys.argv[2]), sys.argv[3])
