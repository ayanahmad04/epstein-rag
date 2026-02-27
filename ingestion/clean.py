import json
import re
import os

FILENAME_RE = re.compile(r'^([A-Za-z0-9_\-\.]+\.txt),?"?(.*)$')

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def clean_dataset(input_path="data/raw.json",
                  output_path="data/cleaned.json"):
    
    with open(input_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    docs = []
    current_file = None
    buffer = []

    def flush():
        nonlocal current_file, buffer
        if current_file and buffer:
            text = clean_text(" ".join(buffer))
            if len(text) >= 100:
                docs.append({
                    "file": current_file,
                    "text": text
                })

    for r in rows:
        line = r.get("text", "").strip()
        if not line or line.lower() == "filename,text":
            continue

        m = FILENAME_RE.match(line)
        if m:
            flush()
            current_file = m.group(1)
            buffer = [m.group(2)]
        else:
            buffer.append(line)

    flush()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print("Cleaned docs:", len(docs))