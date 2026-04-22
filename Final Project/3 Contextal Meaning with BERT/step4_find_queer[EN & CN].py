import re
import json
import jieba
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]  
for word in token_add:
    jieba.add_word(word)
from pathlib import Path

# Load documents
TEXT_DIR = Path("texts")
documents = []
for path in sorted(TEXT_DIR.glob("*.txt")):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if len(text) > 0:
        documents.append({"filename": path.name, "text": text})

print(f"Loaded {len(documents)} documents.\n")

# Search for "queer" and spelling variants:

queer_pattern = re.compile(
    r"(queer|queers|酷儿)", re.IGNORECASE
    )

queer_occurrences = []

for i, doc in enumerate(documents):
    # Split on sentence-ending punctuation
    sentences = re.split(r"[。，！？.;!?]+", doc["text"])               # I add the Chinese punctuations as well
    sentences = [s.strip() for s in sentences if s.strip()]
    
    for sentence in sentences:
        words = list(jieba.cut(sentence))
        match_found = any(queer_pattern.search(word) for word in words)
       
        if match_found:
            queer_occurrences.append({
                "doc_index": i,
                "filename": doc["filename"],
                "sentence": sentence,
                "match": [w for w in words if queer_pattern.search(w)][0], 
            })

# Report
unique_docs = len(set(o["doc_index"] for o in queer_occurrences))
print(f"Found {len(queer_occurrences)} occurrences of 'queer'")
print(f"across {unique_docs} documents.\n")

# Show examples
print("=== Sample 'queer' Occurrences ===\n")
for occ in queer_occurrences[:5]:
    print(f"[{occ['filename']}]")
    print(f"  ...{occ['sentence'][:100]}...")
    print()

# --- Save for next step ---
Path("data").mkdir(exist_ok=True)
with open(Path("data") / "queer_occurrences.json", "w", encoding="utf-8") as f:
    json.dump(queer_occurrences, f, ensure_ascii=False, indent=2)

print(f"Saved {len(queer_occurrences)} occurrences to data/queer_occurrences.json")