import json
import csv
import re
from pathlib import Path
from collections import Counter
from collections import defaultdict

# Load entity mentions from our spaCy work
ENT_PATH = Path.cwd() / "data\\queer_entities_raw.json"

with open(ENT_PATH, "r", encoding="utf-8") as f:
    all_entities = json.load(f)

print("Loaded entity mentions:", len(all_entities))

# Normalization
def normalize_ent(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    
    tokens = []

    for w in text.split():

        if w.isalpha():
            if len(w) > 1:
                tokens.append(w.lower())

        elif re.search(r'[\u4e00-\u9fff]', w):
            if len(w) > 1:
                tokens.append(w)

        else:
            tokens.append(w.lower())

    return " ".join(tokens)


# Noise filtering
def is_noise(ent: str) -> bool:
    if not ent: return True
    ent = ent.strip()
    if len(ent) <= 1 or re.fullmatch(r"[\d\W_]+", ent):   # Remove emoji, number and symbol
        return True
    
    noise_words = {
        "the", "and", "of", "in", "on", "for",
        "with", "is", "are", "was", "were",
        "being", "while","💫顺", "第一", "第二", "第三",
        "一个", "一些", "这个", "那个", "真的", "觉得"}
    if ent in noise_words:
        return True
    return False

clean_entities = []
for e in all_entities:
    norm_txt = normalize_ent(e["entity_text"])
    if not is_noise(norm_txt):
        new_e = e.copy()
        new_e["entity_text"] = norm_txt
        clean_entities.append(new_e)

print("After noise filtering:", len(clean_entities))

# Count
entity_text_counts = Counter(e["entity_text"] for e in clean_entities)

print("Unique entity strings:", len(entity_text_counts))

print("\nTop 20 entity strings:")
for ent_text, n in entity_text_counts.most_common(20):
    print(f"{n:>7}  {ent_text}")

# Label Flitering
KEEP_LABELS = {"PERSON", "GPE"}     #these are the labels I want

counts_by_label = {lab: Counter() for lab in KEEP_LABELS}
examples_by_label = {
    lab: defaultdict(list) for lab in KEEP_LABELS
}

filtered = [                                              #filter for them
    e for e in all_entities
    if e["entity_label"] in KEEP_LABELS
]

print("Filtered entity mentions:", len(filtered))


# Count by label and separate counts by entity type
counts_by_label = {lab: Counter() for lab in KEEP_LABELS}
for e in clean_entities:
    lab = e["entity_label"]
    if lab in KEEP_LABELS:
        txt = e["entity_text"]
        counts_by_label[lab][txt] += 1

        if "sentence" in e:
            if len(examples_by_label[lab][txt]) < 3:
                examples_by_label[lab][txt].append(e["sentence"])  # to save example

for lab in ["PERSON", "GPE"]:
    print(f"\nTop 15 {lab}:")
    for ent_text, n in counts_by_label[lab].most_common(15):
        print(f"{n:>7}  {ent_text}")

# Save the Labels
with open("entity_counts_by_label.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "entity", "count"])
    for lab, counter in counts_by_label.items():
        for ent, cnt in counter.most_common():
            writer.writerow([lab, ent, cnt])

CLEAN_PATH = Path.cwd() / "queer_entities_clean.json"
with open(CLEAN_PATH, "w", encoding="utf-8") as f:
    json.dump(clean_entities, f, ensure_ascii=False, indent=2)

print("Saved cleaned entities to:", CLEAN_PATH.resolve())