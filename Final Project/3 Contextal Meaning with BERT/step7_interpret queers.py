import json
import re
import numpy as np
import jieba                   # Tool for Chinese text segmentation
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]  
for word in token_add:
    jieba.add_word(word)


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

# Configuration
QUEER_EMB_FILE = Path("data") / "queer_embeddings.json"
QUEER_LABEL_FILE = Path("data") / "queer_cluster_labels.json"

TOP_N_TERMS = 30
MAX_FEATURES = 3000
MIN_DF = 2
WINDOW_SIZE = 1
N_SAMPLE = 10

# Custom stopwords
# Baidu Chinese Stopword List + some custom stopwords
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

custom_stopwords = load_stopwords("stopwords_baidu.txt") | {
    "无标题","小红书","红薯","文件", "内容", "数据",
    "一个","一种","特质","或许","马上","标题","带来","喜欢","谢谢","推荐",
    "我要","p1","p2","p3", "哈哈哈哈","笔记"
    }

# Load queer data
with open(QUEER_EMB_FILE, "r", encoding="utf-8") as f:
    queer_data = json.load(f)

with open(QUEER_LABEL_FILE, "r") as f:
    label_data = json.load(f)

queer_metadata = queer_data["metadata"]
queer_labels = label_data["queer_cluster_labels"]

QUEER_SENSES = sorted(list(set(queer_labels)))  # ["A","B"]

print(f"Loaded {len(queer_metadata)} queer instances.\n")


# Load Documents: Build "pseudo-documents"
texts = []
labels = []

for i, meta in enumerate(queer_metadata):
    tokens = meta.get("tokens", None)
    idx = meta.get("target_index", None)
    
    if tokens and idx is not None:
        left = tokens[max(0, idx - WINDOW_SIZE): idx]
        right = tokens[idx + 1: idx + 1 + WINDOW_SIZE]
        context = left + right
        text = " ".join(context)
    else:
        text = meta.get("sentence", "")

    texts.append(text)
    labels.append(queer_labels[i])

labels = np.array(labels)


# Build TF-IDF on the full corpus
# Combine built-in stopwords with custom early modern ones.
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    tokenizer=mixed_tokenizer,
    stop_words=None,
    token_pattern=None
)

tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = np.array(vectorizer.get_feature_names_out())

# remove stopwords
keep_mask = np.array([term not in custom_stopwords for term in feature_names])

tfidf_matrix = tfidf_matrix[:, keep_mask]
feature_names = feature_names[keep_mask]



# Finding the top TF-IDF distcintive words
def get_top_distinctive_terms(tfidf_matrix, labels, target_cluster, feature_names, top_n=10):

    in_cluster = labels == target_cluster
    out_cluster = labels != target_cluster

    if in_cluster.sum() == 0 or out_cluster.sum() == 0:
        return []

    mean_in = tfidf_matrix[in_cluster].mean(axis=0).A1
    mean_out = tfidf_matrix[out_cluster].mean(axis=0).A1
    diff = mean_in - mean_out

    top_indices = diff.argsort()[-top_n:][::-1]
    top_terms = [(feature_names[i], diff[i]) for i in top_indices if diff[i] > 0]

    return top_terms


# Characterize each cluster

print(f"=== QUEER SENSE CHARACTERIZATION (k={len(QUEER_SENSES)}) ===\n")

for sense in QUEER_SENSES:

    mask = labels == sense
    sense_texts = [texts[i] for i in range(len(texts)) if mask[i]]

    print("=" * 70)
    print(f"SENSE {sense} — {len(sense_texts)} instances")
    print("=" * 70)


    # Distinctive TF-IDF terms (cluster vs. rest)
    top_terms = get_top_distinctive_terms(
        tfidf_matrix=tfidf_matrix,
        labels=labels,
        target_cluster=sense,
        feature_names=feature_names,
        top_n=TOP_N_TERMS
    )

    if top_terms:
        formatted_terms = ", ".join([term for term, score in top_terms])
        print(f"Top distinctive terms: {formatted_terms}")
    else:
        print("Top distinctive terms: (none found)")
    
    # Sample filenames
    print("\nSample contexts:")
    shown = 0
    for i, meta in enumerate(queer_metadata):
        if labels[i] == sense:
            sentence = meta.get("sentence", "")
            print(f"  - {sentence}")
            shown += 1
            if shown >= N_SAMPLE:
                break

    print()
