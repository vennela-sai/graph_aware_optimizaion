"""Print mean recall_any@K and recall_all@K (session) for LongMemEval retrieval JSONL."""
import json
import sys
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python summarize_retrieval.py retrieval_gapo.jsonl")
    sys.exit(1)

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
rows = [x for x in rows if "_abs" not in x["question_id"]]

for k in [5, 10, 20]:
    ra = np.mean([x["retrieval_results"]["metrics"]["session"][f"recall_any@{k}"] for x in rows])
    rall = np.mean([x["retrieval_results"]["metrics"]["session"][f"recall_all@{k}"] for x in rows])
    print(f"session recall_any@{k}  = {round(float(ra), 4)}")
    print(f"session recall_all@{k} = {round(float(rall), 4)}")
