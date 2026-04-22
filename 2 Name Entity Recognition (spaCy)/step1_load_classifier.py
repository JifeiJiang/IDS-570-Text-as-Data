from pathlib import Path
import joblib
import json
import jieba
import re
from pathlib import Path

MODEL_DIR = Path.cwd() / "models"

# Load Mixed_tokenizer
token_add = ["性别研究", "交叉性","性少数","gender study", "少数群体","刻板印象","身份认同","gender studies"] # Jieba is not prefrect, I need to add some jargon
for word in token_add:
    jieba.add_word(word)

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

# Load the SAME TF-IDF vectorizer and classifier I trained in 1.5 Step 10 and 11. 
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
clf        = joblib.load(MODEL_DIR / "queer_logreg.joblib")

print("Loaded TF-IDF vectorizer + logistic regression classifier.")


# apply to JSON corpus
with open(Path.cwd() / "data\\new_texts.json", "r", encoding="utf-8") as f:
    records = json.load(f)

texts = [r["text"] for r in records]

X_new = vectorizer.transform(texts)
probs = clf.predict_proba(X_new)[:, 1]
preds = (probs >= 0.50).astype(int)

for r, p, yhat in zip(records, probs, preds):
    r["pred_prob_queer"] = float(p)
    r["pred_queer"] = int(yhat)

print("Classified:", len(records), "documents")
print("Predicted queer (threshold .50):", sum(preds))



# save the classified corpus 
OUT_CLASSIFIED = Path.cwd() / "classified_texts.json"

with open(OUT_CLASSIFIED, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Saved classified dataset:", OUT_CLASSIFIED.resolve())



# create our higher confidence subset
THRESH = 0.70        # Setting the threshold to find higher confident subset
queer_only = [r for r in records if r["pred_prob_queer"] >= THRESH]

OUT_QUEER = Path.cwd() / "queer_texts_for_spacy.json"

with open(OUT_QUEER, "w", encoding="utf-8") as f:
    json.dump(queer_only, f, ensure_ascii=False, indent=2)

print(f"High-confidence queer texts (p >= {THRESH}):", len(queer_only))
print("Saved:", OUT_QUEER.resolve())



# check how this higher threshold compares with the 0.50 one
top5 = sorted(records, key=lambda r: r["pred_prob_queer"], reverse=True)[:5]
for r in top5:
    print(r["doc_id"], r["pred_prob_queer"])