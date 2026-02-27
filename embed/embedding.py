import json
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def embed_chunks(
    input_path="data/chunks.json",
    chroma_dir="chroma_db",
    batch_size=1000,
    reset_db=True  # <-- added control flag
):
    """
    Embeds text chunks into Chroma vector database.

    Args:
        input_path (str): Path to chunked JSON file.
        chroma_dir (str): Directory to persist Chroma DB.
        batch_size (int): Number of chunks per batch.
        reset_db (bool): Whether to delete existing DB before embedding.
    """

    # --------------------------------------------------
    # Reset Chroma DB (Optional)
    # --------------------------------------------------
    if reset_db and Path(chroma_dir).exists():
        print("🧹 Removing existing Chroma DB...")
        shutil.rmtree(chroma_dir)

    # --------------------------------------------------
    # Load chunks
    # --------------------------------------------------
    print("📂 Loading chunks...")
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total = len(chunks)
    print(f"✅ Loaded {total} chunks")

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------
    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model ready")

    # --------------------------------------------------
    # Initialize Chroma DB
    # --------------------------------------------------
    print("📦 Initializing Chroma vector store...")
    db = Chroma(
        collection_name="epstein",
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )

    # --------------------------------------------------
    # Embed in batches
    # --------------------------------------------------
    print(f"🚀 Starting embedding in batches of {batch_size}")

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        print(f"🔹 Embedding chunks {i + 1} → {end} / {total}")

        db.add_texts(
            texts=[c["text"] for c in chunks[i:end]],
            metadatas=[c["metadata"] for c in chunks[i:end]]
        )

    print("🎉 Chroma embedding complete")
    print("📁 Vector DB saved at:", chroma_dir)