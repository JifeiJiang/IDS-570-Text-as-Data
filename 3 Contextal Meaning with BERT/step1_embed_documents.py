import json
from pathlib import Path
import numpy as np
import jieba                   # Tool for Chinese text segmentation
token_add = ["性别研究", "交叉性","性少数","gender study", "刻板印象","身份认同","gender studies"]   # Jieba is not perfrect, I add some jargons 
for word in token_add:
    jieba.add_word(word)
import torch
from transformers import BertModel, BertTokenizer

# Configuration

MODEL_PATH = "bert-base-uncased"
TEXT_DIR = Path("texts")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "doc_embeddings.json"

MAX_LENGTH = 512          # BERT maximum input length
STRIDE = 256              # overlap step
MIN_TOKENS = 2            # ignore small leftover chunks
MIN_CHARS = 1             # ignore extremely short documents

# Device setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    print(f"Loaded the base BERT model from '{model_path}'")

    model.to(device)
    model.eval()
    return tokenizer, model

# Load text files

def load_documents(text_dir, min_chars=1):
    """
    Load .txt documents from a folder.
    Returns a list of dictionaries with filename and text.
    """
    documents = []

    for path in sorted(text_dir.glob("*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if len(text) > min_chars: 
            words = list(jieba.cut(text))
            documents.append({
                "filename": path.name,
                "text": text
            })

    return documents


# Create overlapping token chunks

def chunk_token_ids(token_ids, max_length=512, stride=256, min_tokens=1):
  
    chunk_token_limit = max_length - 2
    chunks = []

    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start:start + chunk_token_limit]
        if len(chunk) >= min_tokens:
            chunks.append(chunk)

    return chunks


# Embed a single chunk

def embed_chunk(chunk_token_ids, tokenizer, model):

    input_ids_list = [tokenizer.cls_token_id] + chunk_token_ids + [tokenizer.sep_token_id]

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Shape: [1, seq_len, hidden_dim]
    last_hidden = outputs.last_hidden_state

    # Exclude [CLS] and [SEP]
    token_embeddings = last_hidden[0, 1:-1, :]

    # Mean pooling across tokens in this chunk
    chunk_embedding = token_embeddings.mean(dim=0).cpu().numpy()

    return chunk_embedding

# Embed a full document

def embed_document(text, tokenizer, model, max_length=512, stride=256, min_tokens=1):
    """
    Embed one long document by:
      1. tokenizing the full document,
      2. splitting into overlapping chunks,
      3. embedding each chunk,
      4. combining chunk embeddings with a length-weighted average.
    """
    
    # Becasue BERT cannot identify Chinese words properly, I use Jieba to segment token first  
    words = list(jieba.cut(text))
    token_ids = []
    for word in words:
        sub_tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        token_ids.extend(ids)

    if len(token_ids) == 0:
        return None

    chunks = chunk_token_ids(
        token_ids,
        max_length=max_length,
        stride=stride,
        min_tokens=min_tokens
    )

    if not chunks:
        return None

    chunk_embeddings = []
    chunk_weights = []

    for chunk in chunks:
        chunk_embedding = embed_chunk(chunk, tokenizer, model)
        chunk_embeddings.append(chunk_embedding)
        chunk_weights.append(len(chunk))  # length-weight the final average

    document_embedding = np.average(
        np.array(chunk_embeddings),
        axis=0,
        weights=np.array(chunk_weights)
    )

    return document_embedding


####
# Main pipeline
def main():
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    documents = load_documents(TEXT_DIR, min_chars=MIN_CHARS)
    print(f"Loaded {len(documents)} documents.\n")

    print("Embedding documents (this will take a while)...\n")

    doc_embeddings = []
    doc_filenames = []

    for i, doc in enumerate(documents, start=1):
        embedding = embed_document(
            doc["text"],
            tokenizer,
            model,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            min_tokens=MIN_TOKENS
        )

        if embedding is not None:
            doc_embeddings.append(embedding.tolist())
            doc_filenames.append(doc["filename"])

        if i % 50 == 0 or i == len(documents):
            print(f"Processed {i}/{len(documents)} documents...")

    if not doc_embeddings:
        print("No embeddings were created.")
        return

    print(f"\nSuccessfully embedded {len(doc_embeddings)} documents.")
    print(f"Embedding dimensionality: {len(doc_embeddings[0])}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    output = {
        "filenames": doc_filenames,
        "embeddings": doc_embeddings
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()