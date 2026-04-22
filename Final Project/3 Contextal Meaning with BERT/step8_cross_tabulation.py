import json
from pathlib import Path
from collections import defaultdict

# Load document cluster assignments(k=2) and labels they use
DATA_DIR = Path("data")
with open(Path("data") / "cluster_assignments.json", "r") as f:
    doc_cluster_data = json.load(f)

doc_cluster_labels = doc_cluster_data["cluster_labels"]
doc_filenames = doc_cluster_data["filenames"]

DOC_CLUSTERS = sorted(list(set(doc_cluster_labels)))

# Build filename -> doc cluster lookup
filename_to_doc_cluster = dict(zip(doc_filenames, doc_cluster_labels))

# Load Queer Data (k=2) and labels they use
with open(DATA_DIR / "queer_embeddings.json", "r", encoding="utf-8") as f:
    queer_data = json.load(f)
with open(DATA_DIR / "queer_cluster_labels.json", "r", encoding="utf-8") as f:
    queer_cluster_data = json.load(f)

queer_metadata = queer_data["metadata"]
queer_cluster_labels = queer_cluster_data["queer_cluster_labels"]

QUEER_SENSES = sorted(list(set(queer_cluster_labels)))


# CROSS-TABULATION: Document Cluster × Queer Sense
cross_tab = defaultdict(int)
doc_cluster_totals = defaultdict(int)
sense_totals = defaultdict(int)

for i, meta in enumerate(queer_metadata):
    fn = meta["filename"]
    queer_sense = queer_cluster_labels[i]
    doc_cluster = filename_to_doc_cluster.get(fn, None)
    
    cross_tab[(doc_cluster, queer_sense)] += 1
    if doc_cluster is not None:
        doc_cluster_totals[doc_cluster] += 1
    sense_totals[queer_sense] += 1

# Print Table 1: Raw Counts
header_width = 15 + (len(QUEER_SENSES) * 14) + 10
print("\n" + "=" * header_width)
print("CROSS-TABULATION: Document Cluster x Queer Sense")
print("=" * header_width)

# Header
print(f"{'Doc Cluster':<15}", end="")
for cs in QUEER_SENSES:
    print(f"{cs:>14}", end="")
print(f"{'Total':>10}")
print("-" * header_width)

# Rows
for dc in DOC_CLUSTERS:
    print(f"{dc:<15}", end="")
    row_total = 0
    for cs in QUEER_SENSES:
        count = cross_tab.get((dc, cs), 0)
        row_total += count
        print(f"{count:>14}", end="")
    print(f"{row_total:>10}")

# Totals row
print("-" * header_width)
print(f"{'Total':<15}", end="")
for cs in QUEER_SENSES:
    print(f"{sense_totals[cs]:>14}", end="")
print(f"{sum(sense_totals.values()):>10}")

# Handle unmatched documents
unmatched = sum(cross_tab.get((None, cs), 0) for cs in QUEER_SENSES)
if unmatched > 0:
    print(f"\n* Note: {unmatched} occurrences were in documents not listed in cluster assignments.")


# PROPORTIONAL VIEW: What % of each document cluster's "queer" usage falls into each sense?
print("\n" + "=" * header_width)
print("PROPORTIONAL VIEW: % of each document cluster's 'queer' by sense")
print("=" * header_width)

for dc in DOC_CLUSTERS:
    total = doc_cluster_totals[dc]
    if total == 0: continue
    
    print(f"{dc:<15}", end="")
    for cs in QUEER_SENSES:
        count = cross_tab.get((dc, cs), 0)
        pct = (count / total) * 100
        print(f"{pct:>13.1f}%", end="")
    print(f"{total:>10}")


# TOP DOCUMENTSL: WHICH DOCUMENTS USE "CREDIT" MOST, AND IN WHICH SENSE?
print("\n" + "=" * 100)
print("TOP DOCUMENTS BY 'queer' FREQUENCY")
print("=" * 100)

# Count per document
doc_queer_counts = defaultdict(lambda: defaultdict(int))
doc_queer_total = defaultdict(int)

for i, meta in enumerate(queer_metadata):
    fn = meta["filename"]
    sense = queer_cluster_labels[i]
    doc_queer_counts[fn][sense] += 1
    doc_queer_total[fn] += 1

# Sort by total count
top_docs = sorted(doc_queer_total.items(), key=lambda x: x[1], reverse=True)[:15]

# Table Header
print(f"{'Filename':<40} {'Total':>6}", end="")
for cs in QUEER_SENSES:
    print(f"{cs:>12}", end="")
print(f"  {'Doc Cluster':>12}")
print("-" * 120)

for fn, total in top_docs:
    dc = filename_to_doc_cluster.get(fn, "N/A")
    print(f"{fn[:38]:<40} {total:>6}", end="")
    for cs in QUEER_SENSES:
        count = doc_queer_counts[fn].get(cs, 0)
        print(f"{count:>12}", end="")
    print(f"  {str(dc):>12}")

# Dominant Documents: DOCUMENTS WITH PREDOMINANTLY ONE SENSE
print("\n" + "=" * 75)
print("DOCUMENTS DOMINATED BY A SINGLE SENSE (>75% of uses)")
print("=" * 75)

for cs in QUEER_SENSES:
    print(f"\n>>> Predominantly {cs}:")
    dominated = []
    for fn, total in doc_queer_total.items():
        if total < 3: continue                          # Skip documents with very few occurrences
        sense_count = doc_queer_counts[fn].get(cs, 0)
        pct = sense_count / total
        if pct >= 0.75:
            dominated.append((fn, total, sense_count, pct))
    
    dominated.sort(key=lambda x: x[1], reverse=True)
    
    if not dominated:
        print("   (none)")
    else:
        for fn, total, sense_count, pct in dominated[:5]:
            dc = filename_to_doc_cluster.get(fn, "N/A")
            print(f"   {fn:<40} {sense_count}/{total} ({pct:.0%}) [Cluster {dc}]")