import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150

# Load queer embeddings
with open(Path("data") / "queer_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

queer_embeddings = np.array(data["embeddings"])
queer_metadata = data["metadata"]

print(f"Loaded {len(queer_embeddings)} 'queer' embeddings.")
print(f"Embedding shape: {queer_embeddings.shape}")

# Find optimal k using silhouette score
if len(queer_embeddings) >= 10:
    k_range = range(2, min(8, len(queer_embeddings) // 2))
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(queer_embeddings)
        sil_scores.append(silhouette_score(queer_embeddings, labels))
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
else:
    best_k = 2
    print(f"Too few occurrences for silhouette analysis. Using k={best_k}.")

# Cluster with k=2
K_QUEER = 2
km_queer = KMeans(n_clusters=K_QUEER, random_state=42, n_init=10)
queer_cluster_labels = km_queer.fit_predict(queer_embeddings)

# Map cluster numbers to custom context names
label_map = {0: "English Cluster?", 1: "Chinese Cluster?"}

print(f"\nQueer clusters (k={K_QUEER}):")
for c in range(K_QUEER):
    count = (queer_cluster_labels == c).sum()
    print(f"  {label_map[c]}: {count} occurrences")

# PCA visualization
pca_queer = PCA(n_components=2)
queer_2d = pca_queer.fit_transform(queer_embeddings)

colors = ["#E74C3C", "#3498DB"]

fig, ax = plt.subplots(figsize=(10, 8))

for c in range(K_QUEER):
    mask = queer_cluster_labels == c
    ax.scatter(
        queer_2d[mask, 0], queer_2d[mask, 1],
        c=colors[c], s=50, alpha=0.6, label=label_map[c],
        edgecolors="black", linewidths=0.3,
    )

ax.set_xlabel(f"PC1 ({pca_queer.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca_queer.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Contextual Embeddings of 'Queer' - Two Senses?")
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("queer_embeddings_pca.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved plots to queer_cluster_selection.png and queer_embeddings_pca.png")

# --- Save cluster labels with custom context names ---
output = {
    "queer_cluster_labels": [label_map[c] for c in queer_cluster_labels],
    "K_QUEER": K_QUEER,
}
with open(Path("data") / "queer_cluster_labels.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Saved cluster labels to data/queer_cluster_labels.json")