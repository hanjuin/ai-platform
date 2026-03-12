import json
import os
import requests
from openai import OpenAI
from datetime import datetime

BASE_URL = "http://localhost:8000"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_token(username:str, password: str) -> str:
    resp = requests.post(f"{BASE_URL}/auth/login", data={
        "username": username,
        "password": password
    })
    resp.raise_for_status()
    return resp.json()["access_token"]

def ask_chat(question: str, token:str) -> dict:
    resp = requests.post(f"{BASE_URL}/chat/", json={"message": question},
                         headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    return resp.json()

def score_answer(question: str, ground_truth: str, predicted_answer: str) -> dict:
    prompt = f"""
        You are evaluating a RAG system's answer quality, Score the answer on two dimensions:
        
        1. **Faithfulness** (0-1): Does the predicted answer contain ONLY information from ground truth?
            Penalize hallucinations or added facts not in the ground truth
        
        2. **Answer Relevance** (0-1): Does the predicted answer actually address the question?
            Penalize irrelevant or off-topic answers.
        
        Question: {question}
        Ground Truth Answer: {ground_truth}
        Predicted Answer: {predicted_answer}
        
        Respond in this exact JSON format:
        {{
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "notes": "brief explanation?"
        }}
    """
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        text={"format": {"type": "json_object"}},
    )

    return json.loads(response.output_text)


def main():
    
    with open("app/evaluation/dataset.json") as f:
        dataset = json.load(f)
        
    USERNAME = "admin"
    PASSWORD = "admin"
    
    token = get_token(USERNAME, PASSWORD)
    
    print(f"Logged in, Running eval on {len(dataset)} questions...\n")
    
    results = []
    
    for i, entry in enumerate(dataset):
        question = entry["question"]
        ground_truth = entry["answer"]
        source_doc = entry["source_document"]
        
        print(f"[{i+1}/{len(dataset)} {question[:60]}...]")
        
        try:
            chat_response = ask_chat(question, token)
            predicted_answer = chat_response["answer"]
            returned_sources = chat_response.get("sources", [])
        except Exception as e:
            print(f"    ERROR calling /chat/: {e}")
            continue
        
        scores = score_answer(question, ground_truth, predicted_answer)
        
        source_retrieved = source_doc in returned_sources
        
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "faithfulness": scores["faithfulness"],
            "answer_relevance": scores["answer_relevance"],
            "source_retrieved": source_retrieved,
            "notes": scores.get("notes", "")
        }
        
        results.append(result)
        
        print(f"    faithfulness={scores['faithfulness']:.2f}"
              f"    relevance={scores['answer_relevance']:.2f}"
              f"    source={'YES' if source_retrieved else 'NO'}")
        
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["answer_relevance"] for r in results) / len(results)
    source_recall = sum(r["source_retrieved"] for r in results) / len(results)
    print(f"Faithfulness:       {avg_faith:.3f}  (higher = less hallucination)")
    print(f"Answer Relevance:   {avg_rel:.3f}  (higher = more on-topic)")
    print(f"Source Recall:      {source_recall:.3f}  (% correct doc was returned)")
    print(f"Total questions:    {len(results)}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"app/evaluation/results/results-{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to app/evaluation/results.json")
    

if __name__ == "__main__":
    main()