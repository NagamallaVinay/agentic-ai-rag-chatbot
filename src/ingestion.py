from pathlib import Path
import json

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")
CHUNKS_PATH = Path("data/chunks.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def extract_pages(pdf_path: Path = PDF_PATH) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "source": pdf_path.name,
            }
        )

    return pages


def create_chunks(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"page-{page['page_number']}-chunk-{chunk_number}",
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "text": chunk_text,
                }
            )

    return chunks


def save_chunks(chunks: list[dict], output_path: Path = CHUNKS_PATH) -> None:
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    pages = extract_pages()
    chunks = create_chunks(pages)
    save_chunks(chunks)

    print(f"Pages with text: {len(pages)}")
    print(f"Total characters: {sum(len(page['text']) for page in pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved chunks to: {CHUNKS_PATH}")
