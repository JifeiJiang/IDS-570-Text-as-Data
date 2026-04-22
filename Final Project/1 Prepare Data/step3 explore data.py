import jieba
from pathlib import Path
from collections import Counter

path = Path("./texts")
all_text = ""

for file_path in path.glob("*.txt"):
    if file_path.is_file():
        all_text += file_path.read_text(encoding="utf-8", errors="replace") + " "

print("Lines:", all_text.count("\n")) 
print("Characters:", len(all_text))
print("\n--- START ---\n")
print(all_text[:100])

tokens = [t.lower().strip() for t in jieba.lcut(all_text) if t.strip()] # Chinese tokenizer (support Chinese and English tokens)
print("Tokens:", len(tokens))
print("Unique tokens:", len(set(tokens)))
counts = Counter(tokens)
print("\nTop 25 tokens:")
for w, c in counts.most_common(10):
    print(f"{w:>12}  {c}")


# Stopwords
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

custom_stopwords = load_stopwords("stopwords_baidu.txt") | {
    "无标题","小红书","红薯","文件", "内容", "数据",
    "一个","一种","特质","或许","马上","标题","带来","喜欢","谢谢","推荐",
    "我要","p1","p2","p3", "哈哈哈哈","笔记"
    }

clean_tokens = [
    t for t in tokens 
    if t not in custom_stopwords and len(t) >= 2
]

print("Tokens (raw):", len(tokens))
print("Tokens (clean):", len(clean_tokens))
print("Unique tokens (clean):", len(set(clean_tokens)))

clean_counts = Counter(clean_tokens)
print("\nTop 25 tokens after cleaning:")

for w, c in clean_counts.most_common(25):
    print(f"{w:>12}  {c}")
