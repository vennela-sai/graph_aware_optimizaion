#!/usr/bin/env python3
"""
LLM-as-judge (same prompts as LongMemEval evaluate_qa.py) using OpenAI or Ollama.
No paid API required if Ollama is running.

Usage:
  set USE_OLLAMA=1
  set OLLAMA_MODEL=llama3.2
  python evaluate_judge.py hyp_gapo.jsonl ..\\LongMemEval\\data\\longmemeval_s_cleaned.json
"""
from __future__ import annotations

import json
import os
import sys
from tqdm import tqdm
from openai import OpenAI

# Prompts aligned with LongMemEval (src/evaluation/evaluate_qa.py)
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a "
                "subset of the information required by the answer, answer no. \n\nQuestion: {}\n\n"
                "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a "
                "subset of the information required by the answer, answer no. In addition, do not penalize "
                "off-by-one errors for the number of days. If the question asks for the number of "
                "days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days "
                "when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\n"
                "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the "
                "response should be considered as correct as long as the updated answer is the required answer."
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response "
                "from a model. Please answer yes if the response satisfies the desired response. "
                "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal information "
                "correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(task)
    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. The model could "
        "say that the information is incomplete, or some other information is given but the asked "
        "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def chat(client: OpenAI, model: str, prompt: str) -> str:
    rsp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    return (rsp.choices[0].message.content or "").strip()


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python evaluate_judge.py hyp_file.jsonl ref_file.json")
        sys.exit(1)
    hyp_file, ref_file = sys.argv[1], sys.argv[2]

    if os.environ.get("OPENAI_API_KEY") and os.environ.get("USE_OLLAMA", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-2024-08-06")
    else:
        client = OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        model = os.environ.get("OLLAMA_JUDGE_MODEL") or os.environ.get(
            "OLLAMA_MODEL", "llama3.2"
        )

    out_path = hyp_file + ".eval-results-local"
    references = json.load(open(ref_file, encoding="utf-8"))
    qid2qdata = {e["question_id"]: e for e in references}
    qid2qtype = {e["question_id"]: e["question_type"] for e in references}
    qtypes = set(qid2qtype.values())
    qtype2acc = {t: [] for t in qtypes}

    try:
        lines = open(hyp_file, encoding="utf-8").readlines()
        hypotheses = [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        hypotheses = json.load(open(hyp_file, encoding="utf-8"))

    logs = []
    with open(out_path, "w", encoding="utf-8") as out_f:
        for entry in tqdm(hypotheses, desc="judge"):
            qid = entry["question_id"]
            if qid not in qid2qtype:
                continue
            qtype = qid2qtype[qid]
            q = qid2qdata[qid]["question"]
            ans = qid2qdata[qid]["answer"]
            hyp = entry["hypothesis"]
            prompt = get_anscheck_prompt(
                qtype, q, ans, hyp, abstention="_abs" in qid
            )
            eval_response = chat(client, model, prompt)
            label = "yes" in eval_response.lower()
            entry["autoeval_label"] = {"model": model, "label": label}
            logs.append(entry)
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            qtype2acc[qtype].append(1 if label else 0)

    acc = sum(1 for x in logs if x["autoeval_label"]["label"]) / max(len(logs), 1)
    print("Accuracy:", round(acc, 4), f"({len(logs)} items)")
    for k, v in qtype2acc.items():
        if v:
            print(f"\t{k}: {round(sum(v) / len(v), 4)} ({len(v)})")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
