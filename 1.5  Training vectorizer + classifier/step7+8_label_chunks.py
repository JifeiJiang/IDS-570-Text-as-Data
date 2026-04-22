from pathlib import Path
import re
import jieba
import json


# Tier definitions
# Tier A: direct spellings / variants of the seed concept
TIER_A = {
    "queer", "queers","酷儿"
}

# Tier B: closely related queer terms
TIER_B = {
    "queer theory","酷儿理论","酷儿性","queerness"
}

# Tier C: "maybe" neighborhood (often adjacent, not always queer-specific)
# I picked some words(not all) from word2vec model
TIER_C = {
    "gender", "性别",
    "woman", "women", "女性", "男性",
    "性少数","lgbtq",
    "巴特勒", "theory",
    "gay", "同性恋", "le", "lesbian", "trans","亚文化", "交叉性",
    "feminism", "女性主义", "女权主义","身份认同","酷酷","边缘",
    "intersectionality", "交叉性","异性恋"
}


# Config
TEXT_DIR = Path("texts")

TARGET_WORDS = 120
MIN_WORDS = 5
MAX_WORDS = 200



# Chunking
def chunk_text(text, target_words=120):
    sentences = re.split(r'([。，,.!?！？])', text)
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = list(jieba.cut(sent))
        if not words:
            continue

        if current_len + len(words) > target_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sent)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# Tokenization
def tokenize(text):
    text = text.lower()
    tokens_en = re.findall(r"[a-zA-Z]+", text)
    tokens_zh = [w.strip() for w in jieba.cut(text) if w.strip()]

    return tokens_en + tokens_zh


# Labeling
def label_text(text):
    tokens = tokenize(text)
    token_set = set(tokens)
    text_lower = text.lower()

    # Tier A / B → CORE
    if (
        token_set & (TIER_A | TIER_B)
        or any(term in text_lower for term in TIER_A | TIER_B)
    ):
        return 1

    # Tier C → MAYBE
    elif (
        token_set & TIER_C
        or any(term in text_lower for term in TIER_C)
    ):
        return 2

    else:
        return 0


# Main pipeline
txt_paths = sorted(TEXT_DIR.glob("*.txt"))
labeled = []

print(f"Processing {len(txt_paths)} files...")

for path in txt_paths:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text, TARGET_WORDS)

    for c in chunks:
        tokens = tokenize(c)
        n_tokens = len(tokens)

        if not (MIN_WORDS <= n_tokens <= MAX_WORDS):
            continue

        label = label_text(c)
        labeled.append((c, label))

print("Total chunks labeled:", len(labeled))


# Save
Path("data").mkdir(exist_ok=True)

with open(Path("data") / "queer_labeled_chunks.json", "w", encoding="utf-8") as f:
    json.dump(labeled, f, ensure_ascii=False)

print("Saved labeled chunks to data/queer_labeled_chunks.json")

# Stats
print("\nLabel distribution:")
print(f"  CORE (1): {sum(1 for _, y in labeled if y == 1)}")
print(f"  NEG  (0): {sum(1 for _, y in labeled if y == 0)}")
print(f"  MAYBE(2): {sum(1 for _, y in labeled if y == 2)}")

# Examples
for label_name, label_val in [("CORE", 1), ("MAYBE", 2), ("NEG", 0)]:
    example = next((text for text, y in labeled if y == label_val), None)
    if example:
        print(f"\n{label_name} example (first 200 chars):")
        print(example[:200])