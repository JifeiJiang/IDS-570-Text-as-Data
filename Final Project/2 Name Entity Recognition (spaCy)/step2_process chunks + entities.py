import spacy
import json
import re
from pathlib import Path
from collections import Counter


# Load the Chinese SpaCy model
nlp = spacy.load("zh_core_web_sm")
print("spaCy model loaded.")


# Manually improve spacy, make it labl more correctly
# Although I know this method is not perfrect, I use it becasue it is efficient for my small corpus (only have 460 texts)
# Also becasue I will focus on person and geography, here I focus on correcting them.
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PERSON", "pattern": "卢凯彤"},
    {"label": "PERSON", "pattern": "福柯"},
    {"label": "PERSON", "pattern": [ {"LOWER": "cocona"}]},
    {"label": "PERSON", "pattern": "哈尔伯斯坦"},
    {"label": "PERSON", "pattern": [ {"LOWER": "lan"},{"LOWER": "liujia"},{"LOWER": "tian"}]},
    {"label": "PERSON", "pattern": "巴特勒"},
    {"label": "PERSON", "pattern": "威廉巴勒斯"},
    {"label": "PERSON", "pattern": [ {"LOWER": "hunter"},{"LOWER": "schafer"}]},
    {"label": "GPE", "pattern": "东亚"},
    {"label": "GPE", "pattern": "亚洲"}
]

ruler.add_patterns(patterns)

# Load data
QUEER_PATH = Path.cwd() / "data\\queer_texts_for_spacy.json"

with open(QUEER_PATH, "r", encoding="utf-8") as f:
    queer_records = json.load(f)

print("Number of queer texts:", len(queer_records))

sample_text = queer_records[0]["text"]

doc = nlp(sample_text)

print("Processed one document.")
print("Number of tokens:", len(doc))


# extract the named entities and inspect them
for ent in doc.ents:
    print(ent.text, "|", ent.label_)

from collections import Counter

Counter([ent.label_ for ent in doc.ents]) 


# scale up to all documents:
all_entities = []

CHUNK_SIZE = 50000  # characters per chunk

for record in queer_records:
    text = record["text"]
    
    # Split text into chunks
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i + CHUNK_SIZE]
        doc = nlp(chunk)
        
        for ent in doc.ents:
            all_entities.append({
                "doc_id": record["doc_id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "sentence": ent.sent.text.strip() 
            })

print("Total entities extracted:", len(all_entities)) 


# Save all the raw entities
OUT_ENTS_RAW = Path.cwd() / "queer_entities_raw.json"

with open(OUT_ENTS_RAW, "w", encoding="utf-8") as f:
    json.dump(all_entities, f, ensure_ascii=False, indent=2)

print("Saved raw entity mentions:", OUT_ENTS_RAW.resolve())


# I SAVE baseline location counts (PERSON, ORE and GPE) for later comparison with trained
KEEP_LABELS = {"PERSON", "ORG","GPE","LOC","NORP"}

def normalize_ent(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

counts_by_label = {lab: Counter() for lab in KEEP_LABELS}

for e in all_entities:
    lab = e["entity_label"]
    if lab in KEEP_LABELS:
        counts_by_label[lab][normalize_ent(e["entity_text"])] += 1

OUT_BASE_COUNTS = Path.cwd() / "queer_locations_counts_base.json"
counts_out = {lab: dict(c.most_common()) for lab, c in counts_by_label.items()}

with open(OUT_BASE_COUNTS, "w", encoding="utf-8") as f:
    json.dump(counts_out, f, ensure_ascii=False, indent=2)

print("Saved baseline counts:", OUT_BASE_COUNTS.resolve())