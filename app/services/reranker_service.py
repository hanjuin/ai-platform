from sentence_transformers import CrossEncoder


class RerankerService:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, chunks: list[dict], content_key: str = "content") -> list[dict]:
        """
        Re-score chunks by reading (query, chunk_text) pairs together.
        Returns chunks sorted by cross-encoder score descending.
        """
        if not chunks:
            return chunks

        pairs = [(query, chunk[content_key]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)


reranker_service = RerankerService()
