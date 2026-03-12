import os
import json
import glob

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

def load_results():
    files = glob.glob(os.path.join(RESULT_DIR, "*.json"))

    def sort_key(f):
        name = os.path.basename(f)
        if name == "result.json":
            return "0"
        return name
    
    return sorted(files, key=sort_key)
    
def compute_metrics(data):
    total = len(data)
    faithfulness = sum(r["faithfulness"] for r in data if r["faithfulness"] is not None) / total
    relevance = sum(r["answer_relevance"] for r in data if r["answer_relevance"] is not None) / total
    source_recall = sum(1 for r in data if r["source_retrieved"]) / total
    
    return {
        "faithfulness" :round(faithfulness, 3),
        "relevance" : round(relevance, 3),
        "source_recall" : round(source_recall, 3),
    }
    
def print_comparison(files):
    all_metrics = []
    
    print(f"\n{'File':<45} {'Faithfulness':>13} {'Relevance':>10} {'Source Recall':>14}")
    print("-"*85)
    
    for path in files:
        with open(path) as f:
            data = json.load(f)
        name = os.path.basename(path)
        m = compute_metrics(data)
        all_metrics.append({"name": name, "data": data, **m})
        print(f"\n{name:<45} {m['faithfulness']:>13} {m['relevance']:>10} {m['source_recall']:>14}")
    
    print_regression(all_metrics)
    
def print_regression(all_metrics):
    if len(all_metrics) < 2:
        return
    
    baseline = all_metrics[0]["data"]
    latest = all_metrics[-1]["data"]
    
    print("\n--- Per-question changes (baseline -> latest) ---")
    for i, (b, l) in enumerate(zip(baseline, latest)):
        delta = round(l["faithfulness"] - b["faithfulness"], 2)
        if delta != 0:
            symbol = "↑" if delta > 0 else "↓"
            print(f"  Q{i+1:02d} {symbol}{abs(delta):.2f}  {b['question'][:70]}")
            
if __name__ == "__main__":
    files = load_results()
    print_comparison(files)