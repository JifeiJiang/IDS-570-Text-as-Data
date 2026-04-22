import json
import re
from pathlib import Path

with open(Path("data") / "queer_occurrences.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Check: how many sentences actually contain "queer" visibly?
check_pattern = re.compile(r"(queer|酷儿|queers)", re.IGNORECASE)

visible = 0
for occ in data[:20]:
    sent = occ["sentence"].lower()
    match = check_pattern.search(sent)

    if match:
        visible += 1
        print(f"FOUND: {sent[:50]}")
    else:
        print(f"NOT VISIBLE: {sent[:50]}...")

print(f"\n{visible}/20 contain 'queer' in the text")