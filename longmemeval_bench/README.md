# LongMemEval-S benchmark runner

## No OpenAI budget? Use **Ollama** (free, local)

1. Install [Ollama](https://ollama.com/) and pull a model (e.g. `ollama pull llama3.2`).
2. In PowerShell:

   ```powershell
   $env:USE_OLLAMA = "1"
   $env:OLLAMA_MODEL = "llama3.2"
   cd c:\Users\moony\Desktop\proj\longmemeval_bench
   python run_benchmark.py --data ..\LongMemEval\data\longmemeval_s_cleaned.json --out outputs --methods gapo --limit 10
   python evaluate_judge.py outputs\hyp_gapo.jsonl ..\LongMemEval\data\longmemeval_s_cleaned.json
   python print_metrics.py outputs\hyp_gapo.jsonl.eval-results-local ..\LongMemEval\data\longmemeval_s_cleaned.json
   ```

3. **Disclose in your paper** that the reader and/or judge was a local model (e.g. Llama 3.2 via Ollama), not GPT-4o — numbers are still reproducible but not directly comparable to LongMemEval leaderboard runs that use OpenAI judges.

If you **do** have `OPENAI_API_KEY`, the code uses the cloud API unless you set `USE_OLLAMA=1`.

---

This folder runs **retrieval + QA** on the official LongMemEval-S cleaned JSON and writes:

- `hyp_<method>.jsonl` — one line per question: `{"question_id", "hypothesis"}` for `evaluate_qa.py`
- `retrieval_<method>.jsonl` — full entries with `retrieval_results.metrics.session` for `print_retrieval_metrics.py`

## Prerequisites

1. **Data** (already in repo if you followed the download):
   - `../LongMemEval/data/longmemeval_s_cleaned.json`

2. **Python** (use project venv):

   ```powershell
   cd c:\Users\moony\Desktop\proj
   .\venv\Scripts\activate
   pip install -r longmemeval_bench\requirements-bench.txt
   ```

3. **OpenAI API key** (for real answers and for official LLM-as-judge):

   ```powershell
   $env:OPENAI_API_KEY = "sk-..."
   ```

4. **LongMemEval evaluation scripts** (cloned at `../LongMemEval`):

   ```powershell
   cd ..\LongMemEval\src\evaluation
   pip install -r ..\..\requirements-lite.txt
   ```

   If `requirements-lite.txt` fails on Windows, install at least: `openai`, `tqdm`, `backoff`, `numpy`.

## 1) Generate hypotheses and retrieval logs

From `longmemeval_bench`:

```powershell
cd c:\Users\moony\Desktop\proj\longmemeval_bench
python run_benchmark.py `
  --data ..\LongMemEval\data\longmemeval_s_cleaned.json `
  --out outputs `
  --methods full_history bm25 dense two_tier gapo `
  --topk 10 `
  --reader-model gpt-4o-mini
```

- Use `--limit 25` while debugging.
- Use `--dry-run` to skip API calls and smoke-test file layout.

**Methods**

| Flag        | Description                                              |
|------------|-----------------------------------------------------------|
| `full_history` | Puts **all** haystack sessions in the reader prompt (long-context baseline). |
| `bm25`     | Lexical BM25 over session documents.                      |
| `dense`    | Cosine similarity with `all-MiniLM-L6-v2`.                |
| `two_tier` | Recent-window + fallback, similarity–degree rank.       |
| `gapo`     | GAPO-style session ranking (sim + graph proxy + entropy). |

## 2) Official QA scoring (LLM judge)

The benchmark uses **`gpt-4o`** as the metric model (see `print_qa_metrics.py` in LongMemEval — it expects the judge model id from that run).

```powershell
cd c:\Users\moony\Desktop\proj\LongMemEval\src\evaluation
$env:OPENAI_API_KEY = "sk-..."
python evaluate_qa.py gpt-4o `
  ..\..\longmemeval_bench\outputs\hyp_gapo.jsonl `
  ..\..\LongMemEval\data\longmemeval_s_cleaned.json
```

This writes `hyp_gapo.jsonl.eval-results-gpt-4o`.

**Macro / per-task table:**

```powershell
python print_qa_metrics.py `
  ..\..\longmemeval_bench\outputs\hyp_gapo.jsonl.eval-results-gpt-4o `
  ..\..\LongMemEval\data\longmemeval_s_cleaned.json
```

Repeat for each `hyp_<method>.jsonl`.

Map printed task names to your paper columns (IE / MSR / TR / KU / Abst) using LongMemEval `question_type` groupings.

## 3) Retrieval metrics (470 non-abstention questions)

```powershell
python print_retrieval_metrics.py ..\..\longmemeval_bench\outputs\retrieval_gapo.jsonl
```

Your paper defines R@\(K\) as “at least one gold evidence session in the top-\(K\)” — that matches **`recall_any@K`** in LongMemEval’s `metrics.session` (not `recall_all@K`, which requires every gold session to appear). From `longmemeval_bench`:

```powershell
cd c:\Users\moony\Desktop\proj\longmemeval_bench
python summarize_retrieval.py outputs\retrieval_gapo.jsonl
```

Paste **`recall_any@5`**, **`recall_any@10`**, **`recall_any@20`** into Table `longmem_ret` unless you intentionally report `recall_all`.

## 4) Paste numbers into `paper_neurocomputing.tex`

Replace the `---` placeholders in Tables `tab:longmem_qa` and `tab:longmem_ret` with your printed metrics.

## Notes

- **Cost:** 500 questions × multiple methods × GPT-4o judge adds up. Use `--limit` until the pipeline is verified.
- **Neo4j:** This runner uses **in-memory** session indexing (same math as the paper’s two-tier/GAPO ideas) so you do not need a live Neo4j for the benchmark. Your production stack still uses Neo4j as in the main codebase.
- If `print_qa_metrics.py` fails on an assertion about the judge model string, either run `evaluate_qa.py` with `gpt-4o` as the first argument or adjust that assertion in a local copy of LongMemEval to match your judge model id.
