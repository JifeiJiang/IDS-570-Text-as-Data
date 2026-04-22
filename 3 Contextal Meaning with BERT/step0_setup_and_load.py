import torch
import numpy as np
import jieba                   # Tool for Chinese text segmentation
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
from pathlib import Path
from transformers import BertTokenizer, BertModel

# Load base BERT
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
print("Using base BERT instead.")

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Load texts
TEXT_DIR = Path("texts")

documents = []
for path in sorted(TEXT_DIR.glob("*.txt")):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    
    words = list(jieba.cut(text)) # load Chinese text segmentation
    documents.append({
         "filename": path.name,  # record the filename
         "text": text,
         "word_count": len(words),
        })

print(f"\nLoaded {len(documents)} documents from {TEXT_DIR}")
print(f"Total words: {sum(d['word_count'] for d in documents):,}")
print(f"Average document length: {np.mean([d['word_count'] for d in documents]):.0f} words")
print(f"Shortest: {min(d['word_count'] for d in documents)} words")
print(f"Longest: {max(d['word_count'] for d in documents)} words")

# --- Preview ---
print("\n=== Sample Documents ===\n")
for doc in documents[:3]:
    print(f"[{doc['filename']}] ({doc['word_count']} words)")
    print(f"  {doc['text'][:200]}...")
    print()

print(f"Setup complete. {len(documents)} documents ready for analysis.")