import json
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict

# Load cleaned entities and original texts
ENT_PATH = Path.cwd() / "data\queer_entities_clean.json"
TEXT_PATH = Path.cwd() / "data\queer_texts_for_spacy.json"

with open(ENT_PATH, "r", encoding="utf-8") as f:
    clean_entities = json.load(f)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    texts = json.load(f)

print("Loaded:", len(clean_entities), "entities,", len(texts), "texts")

# Repeat what I did in step 3
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


# Searching Keywords with entities
KEYWORDS = ["queer", "queers", "酷儿"]
KEEP_LABELS = {"PERSON", "GPE"}

cooccurrence = {
    kw: {lab: Counter() for lab in KEEP_LABELS}
    for kw in KEYWORDS
}

examples = {
    kw: {lab: defaultdict(list) for lab in KEEP_LABELS}
    for kw in KEYWORDS
}

WINDOW_SIZE = 50


for record in texts:
    text = record["text"]
    text_lower = text.lower()

    if not any(kw in text_lower for kw in KEYWORDS):
        continue

    for e in clean_entities:
        ent_text = normalize_ent(e["entity_text"])
        ent_label = e["entity_label"]

        if ent_label not in KEEP_LABELS or is_noise(ent_text):
            continue

        if ent_text in text_lower:
            for kw in KEYWORDS:
                if kw in text_lower:
                    cooccurrence[kw][ent_label][ent_text] += 1
                    
                    if len(examples[kw][ent_label][ent_text]) < 1:
                        examples[kw][ent_label][ent_text].append(text)

# Print the results
for kw in KEYWORDS:
    print(f"\nTop entities near [{kw}]:")
    
    for lab in KEEP_LABELS:
        counter = cooccurrence[kw][lab]
        if not counter:
            continue

        print(f"\n  [{lab}] Top 5:")
        for ent, n in counter.most_common(5):
            print(f"{n:>7}  {ent}")

            for i, sent in enumerate(examples[kw][lab][ent], 1):
                print(f"   ({i}) {sent[:100]}...")


# Visualization
rcParams['font.sans-serif'] = ['DengXian']   # to show Chinese correctly
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

COLOR_MAP = {
    "queer": "#8E44AD",
    "queers": "#8E44AD",
    "酷儿": "#F1C40F"
}

TOP_N = 5 
for kw in KEYWORDS:
    for lab in KEEP_LABELS:
        counter = cooccurrence[kw][lab]

        if not counter:
            continue

        top_items = counter.most_common(TOP_N)

        entities = [x[0] for x in top_items]
        counts = [x[1] for x in top_items]

        plt.figure()

        color = COLOR_MAP.get(kw, "#7F8C8D")
        plt.barh(entities, counts, color=color)

        plt.xlabel("Numbers")
        plt.ylabel("Entity")
        plt.title(f"{kw} related {lab} entities（Top {TOP_N}）")

        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{kw}_{lab}.png", dpi=300, bbox_inches='tight')

        plt.show()