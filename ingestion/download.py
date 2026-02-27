from datasets import load_dataset
import json
import os

def download_dataset(output_path="data/raw.json"):
    os.makedirs("data", exist_ok=True)

    print("Downloading dataset...")
    dataset = load_dataset(
        "teyler/epstein-files-20k",
        split="train"
    )

    docs = []
    for row in dataset:
        docs.append({
            "text": row["text"],
            "file": row.get("file_name", "unknown")
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print("Dataset saved:", output_path)