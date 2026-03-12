from openai import OpenAI
from dotenv import load_dotenv
from typing import Union, List
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmbeddingService:
    
    def generate_embedding(
        self,
        text: Union[str, List[str]]
    ) -> List[List[float]]:
        
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        if isinstance(text, str):
            return [response.data[0].embedding]

        return [item.embedding for item in response.data]


embedding_service = EmbeddingService()