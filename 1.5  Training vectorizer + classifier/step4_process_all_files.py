from pathlib import Path
import jieba                   # Tool for Chinese text segmentation (I use jieba to replace nltk)
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
import re
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

# Load and preprocess all texts
TEXT_DIR = Path("texts")
txt_paths = sorted(TEXT_DIR.glob("*.txt"))

TARGET_WORDS = 120
MIN_WORDS = 1                   # Social media posts are usually short so I don't want to restrict them
MAX_WORDS = 200

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

token_lists = []

print(f"Found {len(txt_paths)} text files.")
print("Building token lists...")

for path in txt_paths:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text, TARGET_WORDS)

    for c in chunks:
        tokens = simple_preprocess(c, deacc=True)
        if MIN_WORDS <= len(tokens) <= MAX_WORDS:
            token_lists.append(tokens)

print("\nTotal tokenized chunks kept:", len(token_lists))


################
# Train Word2Vec (same logic as what we did in week 07)
################

print("\nTraining Word2Vec...")

model = Word2Vec(
    sentences=token_lists,
    vector_size=200,   # dimensionality of word vectors
    window=0,          # context window size
    min_count=1,       # ignore very rare words
    workers=4,         # adjust depending on your machine
    sg=1               # 1 = skip-gram; 0 = CBOW
)

# Save model
Path("models").mkdir(exist_ok=True)
model_path = Path("models") / "w2v_full.bin"
model.save(str(model_path))

print("\nModel saved to:", model_path)