from sentence_transformers import SentenceTransformer
from typing import Union, List

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    def generate_embedding(
        self,
        text: Union[str, List[str]]
    ) -> List[List[float]]:

        embeddings = self.model.encode(
            text,
            normalize_embeddings=True  # improves cosine similarity stability
        )

        if isinstance(text, str):
            return [embeddings.tolist()]

        return embeddings.tolist()


embedding_service = EmbeddingService()