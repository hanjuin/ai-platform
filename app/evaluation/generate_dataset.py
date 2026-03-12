import json
import os
import random
from openai import OpenAI
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5433/aidoc")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_parent_chunks(limit=30):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
                SELECT dc.chunk_id, dc.content, dc.chunk_header, d.filename
                FROM document_chunks dc
                JOIN documents d on dc.document_id = d.document_id
                WHERE dc.parent_chunk_id is NULL
                AND LENGTH(dc.content) > 100
                ORDER BY RANDOM()
                LIMIT %s
                """, (limit,))
    
    rows=cur.fetchall()
    
    cur.close()
    conn.close()
    
    return rows

def generate_qa_pair(chunk_id, content, header, filename):
    prompt = f"""
        You are building a test dataset for a question-answering system.
        
        Given the following document excerpt, generate ONE clear, specific question
        - Can be answered directly from the text
        - Is not trivially obvious (avoid yes/no questions)
        - Tests understanding, not just word matching
        
        Also provide the exact answer from the text (1-3 sentences).
        
        Document: {filename}
        Section: {header or "n/a"}
        
        Text:
        {content[:5000]}
        
        Respond in this exact JSON format:
        {{
            "question": "...",
            "answer": "..."
        }}
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        text={"format": {"type": "json_object"}},
    )

    return json.loads(response.output_text)

def main():
    print("Fetching parent chunks from DB...")
    chunks = get_parent_chunks(limit=30)
    print(f"Found {len(chunks)} chunks, Generating Q&A pairs...")
    
    dataset = []
    
    for chunk_id, content, header, filename in chunks:
        try:
            qa = generate_qa_pair(chunk_id,content,header, filename)
            entry = {
                "question": qa["question"],
                "answer": qa["answer"],
                "source_document": filename,
                "source_chunk_id": str(chunk_id),
                "source_header": header or ""
            }
            dataset.append(entry)
            print(f"    [OK]{filename} - {qa['question'][:60]}...")
        except Exception as e:
            print(f"    [SKIP] chunk {chunk_id}: {e}")
        
    output_path = "app/evaluation/dataset.json"
    with open(output_path,"w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nSaved {len(dataset)} Q&A pairs to {output_path}")
    print("NEXT: Review dataset.json and remove any bad entries before running eval")
    
if __name__ == "__main__":
    main()