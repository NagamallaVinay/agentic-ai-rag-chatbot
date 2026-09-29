import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

from embeddings import LocalEmbedder


CHUNKS_PATH = Path("data/chunks.json")
INDEX_NAME = "agentic-ai-dev"
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    chunks = load_chunks()
    embedder = LocalEmbedder()

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    total_uploaded = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        vectors = []
        for chunk in batch:
            vector = embedder.embed_text(chunk["text"])

            vectors.append(
                {
                    "id": chunk["chunk_id"],
                    "values": vector,
                    "metadata": {
                        "page_number": chunk["page_number"],
                        "source": chunk["source"],
                        "text": chunk["text"],
                    },
                }
            )

        index.upsert(vectors=vectors)
        total_uploaded += len(vectors)

        print(f"Uploaded {total_uploaded}/{len(chunks)} chunks")

    print(f"Finished. Total uploaded: {total_uploaded}")


if __name__ == "__main__":
    main()