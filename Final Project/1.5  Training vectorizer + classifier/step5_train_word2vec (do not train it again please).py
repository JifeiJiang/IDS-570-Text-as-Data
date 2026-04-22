from pathlib import Path
import jieba                   # Tool for Chinese text segmentation (I use jieba to replace nltk)
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
import re
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import os


# Load texts
folder_path = "texts"
documents = []

for filename in os.listdir(folder_path):
    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
        documents.append(f.read())


# Tokenize Texts
def mixed_tokenizer(text):
    tokens = jieba.lcut(text)
    cleaned_tokens = []

    for w in tokens:
        if re.search(r'[\u4e00-\u9fa5]', w):
            if len(w) > 1:
                cleaned_tokens.append(w)
        elif w.isalpha():
            if len(w) > 1:
                cleaned_tokens.append(w.lower())
    return cleaned_tokens

tokenized_sentences = []

for doc in documents:
    sentences = re.split(r'[。！？.!?]', doc)
    for s in sentences:
        tokens = mixed_tokenizer(s)
        if len(tokens) > 2:
            tokenized_sentences.append(tokens)

################
# Train Word2Vec
################

print("\nTraining Word2Vec...")

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,    # dimensionality of word vectors
    window=2,          # context window size (I tried window=5 or 10, results are not good becasue my texts and sentences are both short)
    workers=4,         # adjust depending on your machine
    min_count=2,       # Skip some too rare words
    negative=15,       # Negative Sampling
    epochs=10,         # model cycles
    sg=1               # 1 = skip-gram; 0 = CBOW
)

# Save model
Path("models").mkdir(exist_ok=True)
model_path = Path("models") / "w2v_full_new.bin"
model.save(str(model_path))

print("\nModel saved to:", model_path)