from pathlib import Path
import jieba                   # Tool for Chinese text segmentation'
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
import numpy as np
import joblib
import json


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)
#load the data and use the TF-IDF representations we created
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. Load JSON and data
DATA_DIR = Path("data")

with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)


X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

# Tokenizer text with Jieba
def chinese_tokenizer(text):
    tokens = jieba.lcut(text)
    cleaned_tokens = [w for w in tokens if w.isalpha() and len(w) > 1]
    return cleaned_tokens


# 2. TF–IDF vectorize 
vectorizer = TfidfVectorizer(
    tokenizer=chinese_tokenizer,        # use Chinese tokenizer (jieba)
    token_pattern=None,
    min_df=2,                           # ignore very rare words
    max_df=0.9                          # ignore extremely common words; Explanation [B]
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)


# 3. Train Logistic Regression Model (L2)
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=1
)

clf.fit(X_train, y_train)


# 4. Predictions and Evaluation
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

## 4.1 Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

## 4.2 Classification report (precision/recall/F1)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

## 4.3 ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))

## 4.4 Model sparsity diagnostic: number of non-zero coefficients
coef = clf.coef_
nonzero = np.count_nonzero(coef)
total = coef.size

print("\nModel Sparsity Diagnostic:")
print(f"Non-zero coefficients: {nonzero} / {total}")
print(f"Sparsity (fraction non-zero): {nonzero / total:.4f}")


# 5. Save model
MODEL_DIR = Path.cwd() / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_l2.joblib")
joblib.dump(clf, MODEL_DIR /"queer_logreg_l2.joblib")

print("\nSaved TF-IDF vectorizer and L2 classifier to /models/")


#6. “what the model learned”: Top Words
feature_names = vectorizer.get_feature_names_out()
coef = clf.coef_[0]

# 6.1 Top 15 NEG and CORE
top_pos = np.argsort(coef)[-15:]
top_neg = np.argsort(coef)[:15]

#6.2 Save results
print("\nTop 15 words predicting NEG (L2):")
for i in top_neg:
    print(f"{feature_names[i]:<20} {coef[i]:.4f}")

print("\nTop 15 words predicting CORE (L2):")
for i in reversed(top_pos):
    print(f"{feature_names[i]:<20} {coef[i]:.4f}")