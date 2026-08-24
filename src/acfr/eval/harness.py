import pandas as pd

from acfr.eval.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


def evaluate(predictions, golden, k_values=(1, 3, 5, 10)):
    gmap = {g["query"]: g for g in golden}
    rows = []
    for p in predictions:
        g = gmap[p["query"]]
        pred_ids = [d["doc_id"] for d in p.get("retrieved_docs", [])]
        row = {
            "query": p["query"],
            "answer_match": int(
                p.get("actor_answer", "").strip().lower() == g["reference_answer"].strip().lower()
            ),
            "critic_score": p.get("critic_score", 0.0),
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(pred_ids, g["gold_doc_ids"], k)
            row[f"precision@{k}"] = precision_at_k(pred_ids, g["gold_doc_ids"], k)
            row[f"mrr@{k}"] = mrr_at_k(pred_ids, g["gold_doc_ids"], k)
            row[f"ndcg@{k}"] = ndcg_at_k(pred_ids, g["gold_doc_ids"], k)
        rows.append(row)
    df = pd.DataFrame(rows)
    return {"rows": rows, "summary": df.mean(numeric_only=True).to_dict()}
