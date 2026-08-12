#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) CLI for ResLocoTransformer.

Uses Ollama (local LLM) + ChromaDB (vector store) to answer questions
about the codebase, configs, logs, and research documents.

Commands:
  python scripts/rag.py index            # Index the project files
  python scripts/rag.py query "..."      # One-shot query
  python scripts/rag.py chat             # Interactive chat session
  python scripts/rag.py status           # Show index stats
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / ".rag_db"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"

# File extensions and directories to index
INCLUDE_EXTENSIONS = {
    ".py", ".json", ".md", ".tex", ".sh", ".txt", ".yaml", ".yml", ".xml",
}
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "resloco_transformer.egg-info", ".rag_db",
}
EXCLUDE_FILES = {
    "MUJOCO_LOG.TXT", "crash_log.txt",
}
MAX_FILE_SIZE_KB = 512  # skip files larger than this


def get_ollama():
    try:
        import ollama
        return ollama
    except ImportError:
        print("ERROR: ollama package not installed. Run: pip install ollama")
        sys.exit(1)


def get_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)


def get_collection():
    chromadb = get_chromadb()
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name="resloco",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def collect_files() -> list[Path]:
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix.lower() not in INCLUDE_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_SIZE_KB * 1024:
            continue
        files.append(path)
    return sorted(files)


def embed(texts: list[str]) -> list[list[float]]:
    ollama = get_ollama()
    result = ollama.embed(model=EMBED_MODEL, input=texts)
    return result["embeddings"]


def cmd_index(args):
    collection = get_collection()
    ollama = get_ollama()

    # Verify embed model is available
    try:
        ollama.embed(model=EMBED_MODEL, input=["test"])
    except Exception as e:
        print(f"ERROR: Cannot use embedding model '{EMBED_MODEL}': {e}")
        print(f"  Run: ollama pull {EMBED_MODEL}")
        sys.exit(1)

    files = collect_files()
    print(f"Indexing {len(files)} files from {PROJECT_ROOT}")

    all_ids, all_docs, all_metas, all_embeds = [], [], [], []
    skipped = 0

    for fpath in files:
        rel = str(fpath.relative_to(PROJECT_ROOT))
        try:
            text = fpath.read_text(errors="replace")
        except Exception:
            skipped += 1
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            doc_id = f"{rel}::chunk{i}"
            all_ids.append(doc_id)
            all_docs.append(chunk)
            all_metas.append({"file": rel, "chunk": i})

    print(f"  {len(all_docs)} chunks from {len(files) - skipped} files (skipped {skipped})")
    print(f"  Embedding with {EMBED_MODEL}...")

    # Embed in batches of 64
    batch_size = 64
    embeddings = []
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        embeddings.extend(embed(batch))
        print(f"  Embedded {min(i + batch_size, len(all_docs))}/{len(all_docs)}", end="\r")
    print()

    # Upsert in batches
    for i in range(0, len(all_ids), batch_size):
        collection.upsert(
            ids=all_ids[i : i + batch_size],
            documents=all_docs[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=all_metas[i : i + batch_size],
        )

    print(f"Done. Index stored at {DB_PATH}")
    print(f"Total chunks in index: {collection.count()}")


def cmd_status(args):
    collection = get_collection()
    count = collection.count()
    print(f"Index path : {DB_PATH}")
    print(f"Chunks     : {count}")
    print(f"Embed model: {EMBED_MODEL}")
    print(f"Chat model : {CHAT_MODEL}")
    # Show a few indexed files
    if count > 0:
        sample = collection.get(limit=5)
        files_seen = {m["file"] for m in sample["metadatas"]}
        print(f"Sample files: {', '.join(sorted(files_seen))}")


def retrieve(question: str, n_results: int = 6) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        print("Index is empty. Run: python scripts/rag.py index")
        sys.exit(1)

    q_embed = embed([question])[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "file": meta["file"], "distance": dist})
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for c in chunks:
        context_parts.append(f"--- {c['file']} ---\n{c['text']}")
    context = "\n\n".join(context_parts)

    return f"""You are a helpful assistant for the ResLocoTransformer research project — a PPO-based locomotion RL system for the Unitree Go2 quadruped robot using a transformer-based policy in MuJoCo.

Answer the question using ONLY the context below. If the answer is not in the context, say so clearly. Be concise and precise.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def ask(question: str, history: list[dict] | None = None) -> str:
    ollama = get_ollama()
    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)

    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(model=CHAT_MODEL, messages=messages)
    return response["message"]["content"].strip()


def cmd_query(args):
    question = args.question
    print(f"\nQuery: {question}\n")

    chunks = retrieve(question)
    print(f"Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"  [{c['distance']:.3f}] {c['file']}")
    print()

    answer = ask(question)
    print("Answer:")
    print(textwrap.fill(answer, width=100))


def cmd_chat(args):
    print(f"ResLocoTransformer RAG Chat ({CHAT_MODEL} + {EMBED_MODEL})")
    print("Type 'quit' or 'exit' to end. Type 'sources' after a query to see retrieved chunks.\n")

    history = []
    last_chunks = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "sources":
            if last_chunks:
                print("Sources from last query:")
                for c in last_chunks:
                    print(f"  [{c['distance']:.3f}] {c['file']}")
                    print(f"    {c['text'][:120].replace(chr(10), ' ')}...")
            else:
                print("No previous query.")
            continue

        last_chunks = retrieve(question)
        prompt = build_prompt(question, last_chunks)

        messages = list(history) + [{"role": "user", "content": prompt}]

        ollama = get_ollama()
        try:
            response = ollama.chat(model=CHAT_MODEL, messages=messages)
            answer = response["message"]["content"].strip()
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"  Is the model pulled? Run: ollama pull {CHAT_MODEL}")
            continue

        print(f"\nAssistant: {answer}\n")
        # Keep conversation context (raw question, not the injected prompt)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        # Trim history to last 10 turns
        if len(history) > 20:
            history = history[-20:]


def main():
    parser = argparse.ArgumentParser(
        description="RAG over ResLocoTransformer codebase using Ollama + ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("index", help="Index project files into the vector store")
    sub.add_parser("status", help="Show index statistics")

    p_query = sub.add_parser("query", help="One-shot query")
    p_query.add_argument("question", help="Question to ask")

    sub.add_parser("chat", help="Interactive chat session")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "index": cmd_index,
        "status": cmd_status,
        "query": cmd_query,
        "chat": cmd_chat,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
