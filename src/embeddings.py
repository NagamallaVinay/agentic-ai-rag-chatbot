from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


if __name__ == "__main__":
    embedder = LocalEmbedder()
    vector = embedder.embed_text("What is Agentic AI?")

    print("Embedding model:", MODEL_NAME)
    print("Vector dimension:", len(vector))
    print("First 5 values:", vector[:5])