import json
import jieba                   # Tool for Chinese text segmentation
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
DATA_DIR = Path("data")

with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Separate texts and labels
X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

print("Train size:", len(X_train_texts))
print("Test size :", len(X_test_texts))

# Tokenizer text with Jieba
def mixed_tokenizer(text):
    tokens = jieba.lcut(text)                 # Using Jieba to segment tokens
    cleaned = []

    for w in tokens:
        if re.search(r'[\u4e00-\u9fa5]', w):  # Chinese words should be longer than 1 word
            if len(w) > 1:
                cleaned.append(w)
        elif w.isalpha():                     # English words lowercase, and should be longer than 1 word
            if len(w) > 1:
                cleaned.append(w.lower())

    return cleaned


vectorizer = TfidfVectorizer(
    tokenizer=mixed_tokenizer,          # use Chinese tokenizer (jieba)
    token_pattern=None,
    min_df=1,                           # ignore very rare words, becasue posts are very short, I choose not to ignore too much words.
    max_df=0.9                          # ignore extremely common words
)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

print("TF-IDF matrix shapes:")
print("  Train:", X_train.shape)
print("  Test :", X_test.shape)