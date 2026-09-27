"""Build session-level corpus for LongMemEval (matches LongMemEval/src/retrieval/run_retrieval.py)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def process_item_flat_index(
    data: list, granularity: str, sess_id: str, timestamp: str
) -> Tuple[List[str], List[str], List[str]]:
    """Mirror process_item_flat_index for session or turn granularity."""
    corpus: List[str] = []
    ids: List[str] = []
    timestamps: List[str] = []

    if granularity == "session":
        text = " ".join([interact["content"] for interact in data if interact["role"] == "user"])
        corpus.append(text)
        ids = [sess_id]
        if "answer" in sess_id and all(
            not turn.get("has_answer", False)
            for turn in [x for x in data if x["role"] == "user"]
        ):
            ids = [sess_id.replace("answer", "noans")]
    elif granularity == "turn":
        for i_turn, turn in enumerate(data):
            if turn["role"] == "user":
                corpus.append(turn["content"])
                if "answer" not in sess_id:
                    ids.append(sess_id + "_" + str(i_turn + 1))
                else:
                    assert "has_answer" in turn
                    if turn["has_answer"]:
                        ids.append(sess_id + "_" + str(i_turn + 1))
                    else:
                        ids.append((sess_id + "_" + str(i_turn + 1)).replace("answer", "noans"))
    else:
        raise ValueError(granularity)

    ts = [timestamp for _ in corpus]
    return corpus, ids, ts


def build_corpus_for_entry(entry: Dict[str, Any], granularity: str = "session") -> Tuple[List[str], List[str], List[str]]:
    corpus: List[str] = []
    corpus_ids: List[str] = []
    corpus_timestamps: List[str] = []
    for cur_sess_id, sess_entry, ts in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
        entry["haystack_dates"],
    ):
        cur_items, cur_ids, cur_ts = process_item_flat_index(
            sess_entry, granularity, cur_sess_id, ts
        )
        corpus += cur_items
        corpus_ids += cur_ids
        corpus_timestamps += cur_ts
    return corpus, corpus_ids, corpus_timestamps


def correct_doc_ids(corpus_ids: List[str]) -> List[str]:
    return list({doc_id for doc_id in corpus_ids if "answer" in doc_id})


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
