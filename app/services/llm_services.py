from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a precise question-answering assistant.

Rules you must follow without exception:
1. Answer using ONLY information from the provided context. You may paraphrase, 
   but do not add facts that are not present in the context.
2. Do NOT add facts, details, or elaborations from your training data, even if they seem relevant.
3. If the context does not contain enough information to answer, respond with exactly: "I don't have enough information in the provided documents to answer this."
4. Always cite sources using [Source X] for every claim you make.
5. If only part of the answer is in the context, answer only that part and say the rest is not in the documents."""

def generate_answer(context: str, question: str, history: list[dict] = []) -> str:
    input_messages = []
    
    for msg in history[-6:]:
        input_messages.append(
            {
                "role": msg["role"],
                "content": msg["content"]
            }
        )
        
    input_messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestions: {question}"
    })
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=input_messages
    )

    return response.output_text

def generate_hypothetical(query: str):
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="Write a short 2-3 sentence passage that directly answer the question, written as if its an excerpt from a document",
        input = query
    )
    
    return response.output_text

def generate_answer_stream(
    context: str,
    question: str,
    history: list[dict] = []
):
    input_messages = []
    for msg in history[-6:]:
        input_messages.append({"role": msg["role"], "content": msg["content"]})
    input_messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestions: {question}"
    })
    with client.responses.stream(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=input_messages
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta