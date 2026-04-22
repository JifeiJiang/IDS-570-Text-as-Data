import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150

# Load embeddings
with open(Path("data") / "doc_embeddings.json", "r") as f:
    data = json.load(f)

doc_embeddings = np.array(data["embeddings"])
doc_filenames = data["filenames"]

print(f"Loaded {len(doc_embeddings)} document embeddings.")
print(f"Embedding shape: {doc_embeddings.shape}")

# Find optimal k
print("\nTesting cluster counts...")
k_range = range(2, 12)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(doc_embeddings)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(doc_embeddings, labels))
    print(f"  k={k}: silhouette={silhouettes[-1]:.3f}")

# Plot elbow and silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(list(k_range), inertias, "bo-")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (within-cluster sum of squares)")
ax1.set_title("Elbow Method")
ax1.grid(True, alpha=0.3)

ax2.plot(list(k_range), silhouettes, "ro-")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Analysis")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cluster_selection.png", dpi=150, bbox_inches="tight")
plt.show()

best_k = list(k_range)[np.argmax(silhouettes)]
print(f"\nBest k by silhouette score: {best_k}")

# Fit final clustering
K = best_k
km = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = km.fit_predict(doc_embeddings)

# Map cluster numbers to custom names
cluster_name_map = {
    0: "Individual-related",
    1: "Social-related"
}

print(f"\nClustering with k={K}:")
for cluster_id, count in sorted(Counter(cluster_labels).items()):
    name = cluster_name_map.get(cluster_id, f"Cluster {cluster_id}")
    print(f"  {name}: {count} documents")


# Visualize
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(doc_embeddings)
    method = "UMAP"
except ImportError:
    print("\nUMAP not installed. Using PCA. (For better results: pip install umap-learn)")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(doc_embeddings)
    method = "PCA"

fig, ax = plt.subplots(figsize=(12, 9))
cmap = plt.cm.Set2

for cluster_id in range(K):
    mask = cluster_labels == cluster_id
    label_name = cluster_name_map.get(cluster_id, f"Cluster {cluster_id}")
    ax.scatter(
        reduced[mask, 0], reduced[mask, 1],
        c=[cmap(cluster_id / K)], s=30, alpha=0.6,
        label=f"{label_name} (n={mask.sum()})",
        edgecolors="black", linewidths=0.3,
    )

ax.set_xlabel(f"{method} Dimension 1")
ax.set_ylabel(f"{method} Dimension 2")
ax.set_title(f"Queer Document Clusters ({method}, k={K})")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("document_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved plots to cluster_selection.png and document_clusters.png")

# Save cluster assignments with names
cluster_output = {
    "cluster_labels": [cluster_name_map.get(label, f"Cluster {label}") for label in cluster_labels],
    "filenames": doc_filenames,
    "K": K,
}
with open(Path("data") / "cluster_assignments.json", "w") as f:
    json.dump(cluster_output, f, indent=2)

print("Saved cluster assignments to data/cluster_assignments.json")