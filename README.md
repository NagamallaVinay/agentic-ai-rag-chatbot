# Agentic AI RAG Chatbot

A custom Python-based Retrieval-Augmented Generation (RAG) chatbot built over the Agentic AI eBook knowledge base.

The application implements PDF ingestion, document chunking, semantic embeddings, Pinecone vector retrieval, LangGraph orchestration, Gemini-based answer generation and verification, confidence scoring, grounded refusal for out-of-scope questions, and a Streamlit interface.

## Architecture

```text
Agentic AI PDF
      |
      v
PyPDF Document Extraction
      |
      v
RecursiveCharacterTextSplitter
      |
      |-- Chunk Size: 800 characters
      |-- Chunk Overlap: 100 characters
      |
      v
Sentence Transformers
all-MiniLM-L6-v2
      |
      |-- Embedding Dimension: 384
      |
      v
Pinecone Vector Database
      |
      |-- Similarity Metric: Cosine
      |-- Top-K Retrieval: 5
      |
      v
LangGraph RAG Workflow
      |
      |-- Retrieve
      |-- Generate
      |-- Verify
      |
      v
Streamlit UI



## Execution Commands

git clone https://github.com/NagamallaVinay/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

python src/ingestion.py

python src/index_vectors.py

python -m streamlit run src/app.py