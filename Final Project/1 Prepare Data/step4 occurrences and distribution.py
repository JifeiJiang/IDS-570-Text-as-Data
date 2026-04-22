import os
import re
import jieba

KEYWORDS = r'queers|queer|酷儿'
CORPUS_DIR = './texts'
WINDOW_SIZE = 0          # Only consider that one sentence

def chinese_text(text, KEYWORDS, window):
    sentences = re.split(r'[。，！？.;!?]', text)               # I add the Chinese punctuations as well
    sentences = [s.strip() for s in sentences if s.strip()]

    results = []
    for i, sentence in enumerate(sentences):
        words = list(jieba.cut(sentence))
        match_found = any(re.search(KEYWORDS, word, re.IGNORECASE) for word in words)

        if match_found:
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            context = "。".join(sentences[start:end])
            results.append(context)
    return results

all_occurrences = []
distribution = {}

if not os.path.exists(CORPUS_DIR):
    print(f"Error：cannot find the folder {CORPUS_DIR}")
else:
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(CORPUS_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = chinese_text(content, KEYWORDS, WINDOW_SIZE)
                if matches:
                    all_occurrences.extend(matches)
                    distribution[filename] = len(matches)

# Print the Output
print(f"Report：Keywords:[{KEYWORDS}]")
print(f"1. Total number of occurrences: {len(all_occurrences)}")
print("\n2. Distribution across documents:")
for doc, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {doc:20}: {count} times")
print("\n3. Examples:")
for i, ctx in enumerate(all_occurrences[:5]):
    print(f"   [{i+1}] {ctx}\n" + "-"*20)