from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say you don't know.
Cite sources using [Source X].

Context:
{context}

Question:
{question}

If answer is not in context, say you don't know
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"user", "content": prompt}
        ]
    )

    return response.choices[0].message.content