#!/usr/bin/env python3
"""
LongMemEval-S benchmark driver: retrieval + QA hypotheses + optional official scoring.

Examples:
  set OPENAI_API_KEY=sk-...
  python run_benchmark.py --data ../LongMemEval/data/longmemeval_s_cleaned.json --out outputs --methods bm25 dense two_tier gapo full_history --limit 20

After hypotheses are written:
  cd ../LongMemEval/src/evaluation
  python evaluate_qa.py gpt-4o ..\\..\\longmemeval_bench\\outputs\\hyp_gapo.jsonl ..\\..\\LongMemEval\\data\\longmemeval_s_cleaned.json

  python print_qa_metrics.py hyp_gapo.jsonl.eval-results-gpt-4o ..\\..\\LongMemEval\\data\\longmemeval_s_cleaned.json

Retrieval logs:
  python print_retrieval_metrics.py ..\\..\\longmemeval_bench\\outputs\\retrieval_gapo.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# package imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from longmemeval_io import build_corpus_for_entry, correct_doc_ids, load_dataset
from retrieval_methods import (
    SentenceTransformer,
    bm25_rank,
    dense_rank,
    gapo_session_rank,
    two_tier_session_rank,
)
from metrics import attach_session_metrics
from answer_generation import full_history_answer, generate_answer


METHODS = ("full_history", "bm25", "dense", "two_tier", "gapo")


def rank_sessions(
    method: str,
    query: str,
    corpus: List[str],
    corpus_ids: List[str],
    corpus_timestamps: List[str],
    st_model: SentenceTransformer,
) -> List[int]:
    if method == "bm25":
        return bm25_rank(query, corpus)
    if method == "dense":
        return dense_rank(st_model, query, corpus)
    if method == "two_tier":
        return two_tier_session_rank(st_model, query, corpus, corpus_timestamps)
    if method == "gapo":
        return gapo_session_rank(st_model, query, corpus)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="longmemeval_s_cleaned.json")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=METHODS,
    )
    ap.add_argument("--topk", type=int, default=10, help="Sessions fed to reader")
    ap.add_argument(
        "--reader-model",
        default="",
        help="OpenAI model name if using API; ignored when using Ollama (set OLLAMA_MODEL).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI calls; write placeholder hypotheses",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(str(args.data))
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    print(f"Loaded {len(data)} questions from {args.data}")

    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    for method in args.methods:
        hyp_path = args.out / f"hyp_{method}.jsonl"
        ret_path = args.out / f"retrieval_{method}.jsonl"
        hyp_f = open(hyp_path, "w", encoding="utf-8")
        ret_f = open(ret_path, "w", encoding="utf-8")

        for entry in tqdm(data, desc=method):
            qid = entry["question_id"]
            q = entry["question"]
            corpus, corpus_ids, corpus_ts = build_corpus_for_entry(entry, "session")
            gold = correct_doc_ids(corpus_ids)

            if method == "full_history":
                texts_all = corpus
                ranking = list(range(len(corpus)))  # natural corpus order
                top_texts = texts_all
            else:
                ranking = rank_sessions(
                    method, q, corpus, corpus_ids, corpus_ts, st_model
                )
                top_texts = [corpus[i] for i in ranking[: args.topk]]

            # retrieval log (skip abstention for metrics script convention)
            log_entry = json.loads(json.dumps(entry))
            if "_abs" not in qid:
                attach_session_metrics(
                    log_entry, ranking, corpus_ids, corpus, corpus_ts, gold
                )
            else:
                log_entry.setdefault("retrieval_results", {"metrics": {"session": {}, "turn": {}}})

            ret_f.write(json.dumps(log_entry) + "\n")

            rm = args.reader_model.strip() or None
            if args.dry_run:
                hyp = "[DRY_RUN]"
            elif method == "full_history":
                hyp = full_history_answer(q, top_texts, model=rm)
            else:
                hyp = generate_answer(q, top_texts, model=rm)

            hyp_f.write(
                json.dumps({"question_id": qid, "hypothesis": hyp}, ensure_ascii=False)
                + "\n"
            )

        hyp_f.close()
        ret_f.close()
        print(f"Wrote {hyp_path} and {ret_path}")

    print("\nNext — scoring (pick one):")
    print("  FREE (Ollama): set USE_OLLAMA=1, run: python evaluate_judge.py <hyp>.jsonl <ref.json>")
    print("                 then: python print_metrics.py <hyp>.jsonl.eval-results-local <ref.json>")
    print("  Paid (OpenAI): cd ../LongMemEval/src/evaluation && python evaluate_qa.py gpt-4o ...")
    print("  Retrieval: python print_retrieval_metrics.py <retrieval_*.jsonl>")
    print("             python summarize_retrieval.py <retrieval_*.jsonl>")


if __name__ == "__main__":
    main()
