"""
build_index.py
--------------
Optional helper: rebuild the ChromaDB `medical_knowledge` collection from a
folder of plain-text passages (one passage per file, or one passage per
paragraph inside a file).

You do NOT need to run this if you already have a populated `chroma_store`.
This script exists for the case where you lose the store or want to extend
the knowledge base.

Usage:
    python scripts/build_index.py --source ./corpus --output ./backend/data/chroma_store

Expected source layout:
    corpus/
        radiopaedia/
            pneumonia.txt
            atelectasis.txt
            ...
        rsna/
            ...
        nih_bookshelf/
            ...

Each .txt file becomes one or more passages (split on blank lines).
The folder name is captured as the `source` metadata, and the filename
as the `topic`.
"""

import os
import re
import argparse
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build/rebuild the MedLens ChromaDB index.")
    p.add_argument(
        "--source",
        required=True,
        help="Path to a folder containing your corpus (subfolders = source names).",
    )
    p.add_argument(
        "--output",
        default="./backend/data/chroma_store",
        help="Where to write the persistent ChromaDB (default: backend/data/chroma_store).",
    )
    p.add_argument(
        "--collection",
        default="medical_knowledge",
        help="Collection name (default: medical_knowledge).",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=80,
        help="Skip passages shorter than this many characters (default: 80).",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before re-indexing.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Passage extraction
# -----------------------------------------------------------------------------

def extract_passages(source_root: Path, min_len: int):
    """
    Walks the source folder and yields (passage_text, source_name, topic).

    source_name comes from the first sub-folder under source_root.
    topic comes from the .txt filename.
    """
    for source_dir in sorted(source_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name
        for txt_path in sorted(source_dir.rglob("*.txt")):
            topic = txt_path.stem.replace("_", " ").title()
            content = txt_path.read_text(encoding="utf-8", errors="ignore")
            # Split on blank lines into paragraph-sized passages
            paragraphs = re.split(r"\n\s*\n", content)
            for para in paragraphs:
                para = para.strip()
                if len(para) >= min_len:
                    yield para, source_name, topic


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    source_root = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not source_root.is_dir():
        raise SystemExit(f"Source folder not found: {source_root}")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[build_index] Source : {source_root}")
    print(f"[build_index] Output : {output_path}")
    print(f"[build_index] Collection : {args.collection}")
    print()

    # Encoder
    print("[build_index] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ChromaDB
    client = chromadb.PersistentClient(path=str(output_path))
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print("[build_index] Existing collection deleted.")
        except Exception:
            pass
    collection = client.get_or_create_collection(name=args.collection)

    # Extract + embed
    print("[build_index] Extracting passages...")
    passages, sources, topics, ids = [], [], [], []
    for text, source_name, topic in extract_passages(source_root, args.min_len):
        passages.append(text)
        sources.append(source_name)
        topics.append(topic)
        ids.append(f"{source_name}_{uuid.uuid4().hex[:10]}")

    if not passages:
        raise SystemExit("[build_index] No passages found. Check --source path.")

    print(f"[build_index] Embedding {len(passages)} passages...")
    embeddings = encoder.encode(passages, show_progress_bar=True).tolist()

    print("[build_index] Writing to ChromaDB...")
    collection.add(
        ids=ids,
        documents=passages,
        embeddings=embeddings,
        metadatas=[
            {"source": s, "topic": t}
            for s, t in zip(sources, topics)
        ],
    )

    print(f"[build_index] Done. Collection now contains {collection.count()} passages.")


if __name__ == "__main__":
    main()
