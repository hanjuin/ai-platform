from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are an assistant
Use ONLY the context below to answer the question

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