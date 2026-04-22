import torch
import json
import jieba                   # Tool for Chinese text segmentation
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]  
for word in token_add:
    jieba.add_word(word)  
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel

# Load mode# I used bert-base-uncased and found it just simply separated Chinese and English texts
# Thus, I use the mix-lngustic model
MODEL_PATH = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
print(f"Using base BERT")


model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load queer occurrences:
with open(Path("data") / "queer_occurrences.json", "r", encoding="utf-8") as f:
    queer_occurrences = json.load(f)

print(f"Loaded {len(queer_occurrences)} queer occurrences.\n")


# Word embedding extraction function. See [1] below
def get_word_embedding(sentence, target_word, tokenizer, model):
    
    # Becasue BERT cannot identify Chinese words properly, I use Jieba to segment token first
    words = list(jieba.cut(sentence))
    inputs = tokenizer(
        words, 
        is_split_into_words=True,
        return_tensors="pt", 
        truncation=True, 
        max_length=512)
    
    inputs = inputs.to(device)
    
    word_ids = inputs.word_ids(batch_index=0)
    target_indices = [
        i for i, word_id in enumerate(word_ids)
        if word_id is not None and words[word_id] == target_word
    ]
    if not target_indices:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state[0].cpu()
    word_embedding = hidden_states[target_indices].mean(dim=0).numpy()

    return word_embedding


# Extract embeddings:
print("Extracting contextual embeddings for 'queer'...\n")

queer_embeddings = []
queer_metadata = []

for i, occ in enumerate(queer_occurrences):
    emb = get_word_embedding(occ["sentence"], occ["match"], tokenizer, model)

    if emb is not None:
        queer_embeddings.append(emb.tolist())
        queer_metadata.append(occ)

    if (i + 1) % 25 == 0:
        print(f"  Processed {i + 1}/{len(queer_occurrences)}...")

print(f"\nExtracted {len(queer_embeddings)} embeddings for 'queer'.")

# Save
output = {
    "embeddings": queer_embeddings,
    "metadata": queer_metadata,
}
with open(Path("data") / "queer_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print("Saved to data/queer_embeddings.json")