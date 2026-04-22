import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load data and labels
with open("data/EN_embeddings_and_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X = np.array(data["embeddings"])
y_labels = np.array(data["queer_cluster_labels_new"])


# 2. Encode labels
le = LabelEncoder()
y = le.fit_transform(y_labels)

for i, name in enumerate(le.classes_):
    print(f"Label {i} -> {name}")


# 3. Standarization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. Weak lableling (per class 12)
## My trainset only use 12 per class beacasue the English cluster is too small
### Acadmeic Theory only has 25 occurrences and Popular Culture only has 59 occurrences
np.random.seed(42)
train_idx = []
for cls in np.unique(y):
    idx = np.where(y == cls)[0]

    if len(idx) < 12:
        raise ValueError(f"Class {cls} has only {len(idx)} samples")

    sampled = np.random.choice(idx, 12, replace=False)
    train_idx.extend(sampled)

train_idx = np.array(train_idx)

# train set
X_train = X_scaled[train_idx]
y_train = y[train_idx]

# test set
mask = np.ones(len(y), dtype=bool)
mask[train_idx] = False

X_test = X_scaled[mask]
y_test = y[mask]

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


# 5. Logistic Regression (L1)
clf = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=2000
    )

clf.fit(X_train, y_train)


# 6. Evaluate Performances
## 6.1 Predictions and Evaluation
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

## 6.2 Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

## 6.3 Classification report (precision/recall/F1)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

## 6.4 ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))

## 6.5 Model sparsity diagnostic: number of non-zero coefficients
coef = clf.coef_
nonzero = np.count_nonzero(coef)
total = coef.size

print("\nModel Sparsity Diagnostic:")
print(f"Non-zero coefficients: {nonzero} / {total}")
print(f"Sparsity (fraction non-zero): {nonzero / total:.4f}")
