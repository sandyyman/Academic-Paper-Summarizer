from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingHandler:
    def __init__(self, model_name: str = "BAAI/bge-base-en"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
