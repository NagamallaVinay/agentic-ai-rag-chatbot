import os
import re

from dotenv import load_dotenv
from pinecone import Pinecone

from src.embeddings import LocalEmbedder


load_dotenv()

INDEX_NAME = "agentic-ai-dev"
RETRIEVE_K = 30
FINAL_K = 10

IMPORTANT_TERMS = {
    "challenge",
    "challenges",
    "consideration",
    "considerations",
    "implementation",
    "implementing",
    "governance",
    "security",
    "risk",
    "risks",
    "testing",
    "validation",
    "vulnerability",
    "vulnerabilities",
    "data",
    "infrastructure",
    "readiness",
    "strategy",
    "talent",
    "privacy",
}


def tokenize(text: str) -> set[str]:
    return set(
        re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    )


def keyword_score(query: str, text: str) -> float:
    query_words = tokenize(query)
    text_words = tokenize(text)

    if not query_words:
        return 0.0

    overlap = query_words.intersection(text_words)

    return len(overlap) / len(query_words)


def important_term_score(text: str) -> float:
    words = tokenize(text)

    matches = words.intersection(IMPORTANT_TERMS)

    if not matches:
        return 0.0

    return min(len(matches) / 5.0, 1.0)


def retrieve(query: str) -> list[dict]:
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    embedder = LocalEmbedder()
    query_vector = embedder.embed_text(query)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    result = index.query(
        vector=query_vector,
        top_k=RETRIEVE_K,
        include_metadata=True,
    )

    candidates = []

    for match in result.matches:
        text = match.metadata.get("text", "")

        if not text:
            continue

        semantic = float(match.score)
        lexical = keyword_score(query, text)
        important = important_term_score(text)

        combined = (
            0.55 * semantic
            + 0.25 * lexical
            + 0.20 * important
        )

        candidates.append(
            {
                "id": match.id,
                "score": combined,
                "semantic_score": semantic,
                "keyword_score": lexical,
                "important_term_score": important,
                "page_number": match.metadata.get("page_number"),
                "source": match.metadata.get("source"),
                "text": text,
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[:FINAL_K]


if __name__ == "__main__":
    query = (
        "What are the key challenges and considerations "
        "when implementing Agentic AI?"
    )

    matches = retrieve(query)

    print(f"Query: {query}")
    print(f"Retrieved: {len(matches)} chunks")

    for i, match in enumerate(matches, start=1):
        print(f"\n--- Result {i} ---")
        print("Page:", match["page_number"])
        print("Score:", round(match["score"], 4))
        print("Text:", match["text"][:700])