import json
from pathlib import Path

# Point to the folder with our new Post Spring Break Texts (or whatever you named it)
NEW_TEXTS_DIR = Path.cwd() / "texts"


# Read all .txt files
paths = sorted(NEW_TEXTS_DIR.glob("*.txt"))
print("Found .txt files:", len(paths))

# How the JSON file will be organized
records = []
for p in paths:
    text = p.read_text(encoding="utf-8", errors="ignore")
    records.append({
        "doc_id": p.stem,           # filename without extension
        "filename": p.name,         # full filename
        "text": text                # raw text
    })

# Quick check
print("First record keys:", records[0].keys())
print("First doc_id:", records[0]["doc_id"])
print("First text snippet:", records[0]["text"][:200])

# Save the corpus as JSON
OUT_JSON = Path.cwd() / "new_texts.json"

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_JSON)