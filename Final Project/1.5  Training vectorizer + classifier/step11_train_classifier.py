from pathlib import Path
import re
import jieba                   # Tool for Chinese text segmentation
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
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


# 3. Train Logistic Regression Model
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=1
)

clf.fit(X_train, y_train)


#test set predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# evaluation components
###classification report that summarizes how reliable positive predictions are 
print("\nClassification report:")
print(classification_report(y_test, y_pred))

###ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))


#Saving TF-IDF vectorizer and classifier
from pathlib import Path
import joblib

MODEL_DIR = Path.cwd() / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
joblib.dump(clf, MODEL_DIR / "queer_logreg.joblib")

print("Saved TF-IDF vectorizer and classifier to /models/")