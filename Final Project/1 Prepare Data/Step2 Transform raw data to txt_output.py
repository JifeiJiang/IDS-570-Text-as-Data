import csv
import os
import csv
import os

csv_file = "raw data.csv"

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = list(csv.reader(f))

output_dir = "texts"
os.makedirs(output_dir, exist_ok=True)

for i, row in enumerate(reader, start=1):
    a_col_text = row[0] if len(row) > 0 else ""
    b_col = row[1] if len(row) > 1 else ""
    c_col_text = row[2] if len(row) > 2 else ""

# Skip the lines are related to B=1 (I find they are related to Qoo drinks), or C="" (failed to crawl any content) 
    if b_col == "1" or c_col_text == "":
        continue 

    txt_content = f"{a_col_text}\n{c_col_text}"

    file_path = os.path.join(output_dir, f"{i}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    print(f"save document: {file_path}")