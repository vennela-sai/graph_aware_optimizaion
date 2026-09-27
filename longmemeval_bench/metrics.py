"""Retrieval metrics (from LongMemEval/src/retrieval/eval_utils.py)."""
import numpy as np
from typing import List


def dcg(relevances, k):
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0


def ndcg(rankings, correct_docs, corpus_ids, k=10):
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(sorted_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(doc in recalled_docs for doc in correct_docs))
    recall_all = float(all(doc in recalled_docs for doc in correct_docs))
    ndcg_score = ndcg(rankings, correct_docs, corpus_ids, k)
    return recall_any, recall_all, ndcg_score


def attach_session_metrics(
    entry: dict,
    rankings: List[int],
    corpus_ids: List[str],
    corpus_texts: List[str],
    corpus_ts: List[str],
    correct_docs: List[str],
) -> dict:
    """Mutate entry with retrieval_results.metrics.session for official print_retrieval_metrics.py."""
    rr = entry.setdefault("retrieval_results", {})
    rr.setdefault("query", entry["question"])
    rr["ranked_items"] = [
        {
            "corpus_id": corpus_ids[i],
            "text": corpus_texts[i][:2000],
            "timestamp": corpus_ts[i],
        }
        for i in rankings
    ]
    m = rr.setdefault("metrics", {}).setdefault("session", {})
    for k in [1, 3, 5, 10, 30, 50]:
        ra, rr_all, nd = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
        m[f"recall_any@{k}"] = ra
        m[f"recall_all@{k}"] = rr_all
        m[f"ndcg_any@{k}"] = nd
    rr.setdefault("metrics", {}).setdefault("turn", {})
    return entry
