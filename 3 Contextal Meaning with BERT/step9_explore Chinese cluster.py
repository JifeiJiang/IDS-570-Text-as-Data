# Becasue the results of BERT word-level embedding are too lingusitically dichotomic,
# In step 9, I re-run BERT cluster and interpret of the "Chinese Clusters?" and "English Clusters?" separatedly
# My goal is to understand what actually exist inside two clusters

import json
import re
import numpy as np
import jieba
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]
for word in token_add:
    jieba.add_word(word)
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.rcParams["figure.dpi"] = 150

# 1. Load embeddings
with open(Path("data") / "queer_embeddings.json", "r", encoding="utf-8") as f:
    emb_data = json.load(f)

X = np.array(emb_data["embeddings"])
metadata = emb_data["metadata"]

with open(Path("data") / "queer_cluster_labels.json", "r", encoding="utf-8") as f:
    label_data = json.load(f)

old_labels = np.array(label_data["queer_cluster_labels"])

# 2. Extracting data from the "Chinese cluster?"
chinese_idx = np.where(old_labels == "Chinese Cluster?")[0]
X_chinese = X[chinese_idx]
metadata_chinese = [metadata[i] for i in chinese_idx]

print(f"Filtered {len(X_chinese)} Chinese occurrences from the main dataset.\n")

# 3. Re-clustering queer in that cluster
# 3.1 Find optimal k using silhouette score
k_range = range(2, min(8, len(X_chinese) // 2))
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_chinese)
    sil_scores.append(silhouette_score(X_chinese, labels))
    print(f"  k={k}: silhouette={sil_scores[-1]:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), sil_scores, "ro-")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Optimal Clusters for 'queer' Embeddings")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("queer_cluster_selection.png", dpi=150, bbox_inches="tight")
plt.show()

best_k = list(k_range)[np.argmax(sil_scores)]
print(f"\nBest k by silhouette: {best_k}")

# 3.2 Cluster with k=2
K_QUEER = 3
km_queer = KMeans(n_clusters=K_QUEER, random_state=42, n_init=10)
new_CN_labels = km_queer.fit_predict(X_chinese)

# 3.3 Map cluster numbers to custom context names
label_map = {0: "Social Category", 1: "Public Knowledge", 2:"Subjective Experience"}
mapped_CN_labels = [label_map[c] for c in new_CN_labels]

print(f"\nNew Chinese Queer clusters (k={K_QUEER}):")
for c in range(K_QUEER):
    count = (new_CN_labels == c).sum()
    print(f"  {label_map[c]}: {count} occurrences")

# 3.4 PCA visualization
pca_queer = PCA(n_components=2)
queer_2d = pca_queer.fit_transform(X_chinese)

colors = ["#E74C3C", "#3498DB", "#2ECC71"]  # Red, Blue, Green

fig, ax = plt.subplots(figsize=(10, 8))

for c in range(K_QUEER):
    mask = new_CN_labels == c
    ax.scatter(
        queer_2d[mask, 0], queer_2d[mask, 1],
        c=colors[c], s=50, alpha=0.6, label=label_map[c],
        edgecolors="black", linewidths=0.3,
    )

ax.set_xlabel(f"PC1 ({pca_queer.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca_queer.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Contextual Embeddings of Chinese Cluster 'Queer' within Text")
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("Chinese_queer_embeddings_pca.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved plots to Chinese_queer_cluster_selection.png and Chinese_queer_embeddings_pca.png")


# 4. Interpret Queers
# 4.1 Process Texts and TF-IDF 
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


texts = [meta.get("sentence", "") for meta in metadata_chinese]
labels_array = np.array(mapped_CN_labels)

vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    tokenizer=mixed_tokenizer,
    token_pattern=None
)

tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = np.array(vectorizer.get_feature_names_out())


# 4.2 Remove stopwords after vectorizer to make resutls more readable (not impact measuring)
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

custom_stopwords = load_stopwords("stopwords_baidu.txt") | {
    "无标题","小红书","红薯","文件", "内容", "数据",
    "一个","一种","特质","或许","马上","标题","带来","喜欢","谢谢","推荐","不想","范畴",
    "我要","p1","p2","p3", "哈哈哈哈","笔记","虽非","绝非","对方","是否是","一步","关乎","误以为","之外"
    "某种意义","误认为","确认","该词", "最广","笔记"

    }
keep_mask = np.array([term not in custom_stopwords for term in feature_names])
tfidf_matrix = tfidf_matrix[:, keep_mask]
feature_names = feature_names[keep_mask]


# 4.3 Find the Top_distinctive terms
def get_top_distinctive_terms(tfidf_matrix, labels, target_cluster, feature_names, top_n=15):
    in_cluster = labels == target_cluster
    out_cluster = labels != target_cluster
    
    if in_cluster.sum() == 0 or out_cluster.sum() == 0:
        return []
        
    mean_in = tfidf_matrix[in_cluster].mean(axis=0).A1
    mean_out = tfidf_matrix[out_cluster].mean(axis=0).A1
    diff = mean_in - mean_out
    
    top_indices = diff.argsort()[-top_n:][::-1]
    return [(feature_names[i], diff[i]) for i in top_indices if diff[i] > 0]


# 4.4 Characterize each cluster
QUEER_SENSES = ["Social Category", "Public Knowledge","Subjective Experience"]
print(f"=== QUEER SENSE CHARACTERIZATION (k={len(QUEER_SENSES)}) ===\n")

for sense in QUEER_SENSES:
    print("=" * 70)
    print(f"SENSE: {sense}")
    print("=" * 70)

    # Distinctive TF-IDF terms (cluster vs. rest)
    top_terms = get_top_distinctive_terms(
        tfidf_matrix=tfidf_matrix,
        labels=labels_array,
        target_cluster=sense,
        feature_names=feature_names,
        top_n=30
    )

    if top_terms:
        formatted_terms = ", ".join([term for term, score in top_terms])
        print(f"Top distinctive terms: {formatted_terms}")
    else:
        print("Top distinctive terms: (none found)")
    
    # Sample filenames
    print("\nSample contexts:")
    mask = labels_array == sense
    sense_texts = [texts[i] for i in range(len(texts)) if mask[i]]
    for i in range(min(5, len(sense_texts))):
        print(f"  - {sense_texts[i][:100]}...")
    print()

# 5. Save the Results and DATA
output = {
    "embeddings": X_chinese.tolist(),
    "queer_cluster_labels_new": mapped_CN_labels,
    "metadata": metadata_chinese
}

with open(Path("data") / "CN_embeddings_and_labels.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print("Saved internal Chinese sub-clusters to data/CN_embeddings_and_labels..json")