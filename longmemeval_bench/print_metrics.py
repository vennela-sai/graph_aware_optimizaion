#!/usr/bin/env python3
"""Aggregate QA accuracy from evaluate_judge output (any judge model)."""
import json
import sys
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python print_metrics.py hyp_file.eval-results-local ref_file.json")
    sys.exit(1)

in_file, ref_file = sys.argv[1], sys.argv[2]
in_data = [json.loads(line) for line in open(in_file, encoding="utf-8")]
ref_data = {x["question_id"]: x for x in json.load(open(ref_file, encoding="utf-8"))}

from collections import defaultdict

type2acc = defaultdict(list)
all_acc = []
abstention_acc = []

for entry in in_data:
    ref_entry = ref_data[entry["question_id"]]
    t = ref_entry["question_type"]
    ok = 1 if entry["autoeval_label"]["label"] else 0
    type2acc[t].append(ok)
    all_acc.append(ok)
    if "_abs" in entry["question_id"]:
        abstention_acc.append(ok)

print("\nBy task:")
for k in sorted(type2acc.keys()):
    v = type2acc[k]
    if not v:
        continue
    m = float(np.mean(v))
    print(f"\t{k}: {round(m, 4)} ({len(v)})")

main_types = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]
macro = np.mean([np.mean(type2acc[t]) for t in main_types if type2acc[t]])
print("\nMacro (mean of 6 main types):", round(float(macro), 4))
print("Overall (all questions):", round(float(np.mean(all_acc)), 4))
if abstention_acc:
    print("Abstention:", round(float(np.mean(abstention_acc)), 4), f"({len(abstention_acc)})")
