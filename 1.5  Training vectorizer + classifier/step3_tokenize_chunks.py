from pathlib import Path
import jieba                   # Tool for Chinese text segmentation (I use jieba to replace nltk)
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
import re
from gensim.utils import simple_preprocess

TEXT_DIR = Path("texts")
txt_paths = sorted(TEXT_DIR.glob("*.txt"))

sample_path = txt_paths[0]

with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Step 2 logic: sentences -> chunks (~120 words)
sentences = re.split(r'([。，,.!?！？])', text)              # I add Chinese punctuations
entences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]

print("File:", sample_path)
print("Number of sentences:", len(sentences))
print()

# Group sentences into chunks of ~120 words
TARGET_WORDS = 120
chunks = []
current = []
current_len = 0

for sent in sentences:
   sent = sent.strip()
   if not sent:
        continue
   
   words = list(jieba.cut(sent))       # Introduce Jieba to tokenize texts
   
   # If adding this sentence would exceed the target, finalize the chunk
   if current_len + len(words) > TARGET_WORDS and current:
        chunks.append(" ".join(current))
        current = []
        current_len = 0
        
   current.append(sent)
   current_len += len(words)

# Add any leftover sentences
if current:
    chunks.append(" ".join(current))

# Step 3: now we tokenize each chunk
token_lists = [simple_preprocess(c, deacc=True) for c in chunks]

print("File:", sample_path)
print("Chunks (strings):", len(chunks))
print("Chunks (token lists):", len(token_lists))

print("\n--- Token preview (first 60 tokens of first chunk) ---")
print(token_lists[0][:60])

print("\n--- Token count of first chunk ---")
print(len(token_lists[0]))