import cohere
from dotenv import load_dotenv
import os
load_dotenv()

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

class RerankerService:
    def rerank(self, query: str, chunks: list[dict], content_key: str = "content") -> list[dict]:
        """
        Re-score chunks by reading (query, chunk_text) pairs together.
        Returns chunks sorted by cross-encoder score descending.
        """
        if not chunks:
            return chunks
        
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[chunk[content_key] for chunk in chunks]
        )

        for result in response.results:
            chunks[result.index]["rerank_score"] = float(result.relevance_score)

        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)


reranker_service = RerankerService()
