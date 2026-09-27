# Graph-Based Persistent Memory for Conversational AI

A **wrapper-agnostic**, graph-based persistent memory layer for large-scale conversational AI. Multiple LLM providers (Gemini, Qwen, Claude, DeepSeek, Codestral) share a single knowledge graph backed by **Neo4j**, with semantic, entity, and temporal links. Supports **GAPO** (Graph-Aware Prompt Optimization), memory compression, integrity checking, and multi-persona responses.

---

## Features

- **Unified graph memory**: Conversation turns as nodes; SEMANTIC, ENTITY, and TEMPORAL edges.
- **Wrapper-agnostic**: One memory graph for any LLM (Aikipedia/Gemini, Qwen, Claude, etc.).
- **GAPO**: Optimized retrieval via Score = α·similarity + β·PageRank + γ·informativeness.
- **Multi-persona**: Ask as Gemini, Qwen, Claude, DeepSeek, or Codestral with shared memory.
- **Two-tier retrieval**: Fast path over recent nodes; fallback to full-graph search when needed.
- **Memory compression**: Louvain clustering + summarization for large graphs (e.g. 10k+ nodes).
- **Integrity analysis**: Contradiction and hallucination checks against retrieved memory.
- **Export**: Cytoscape JSON, Neo4j Cypher scripts, GEXF for Gephi.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM Wrappers (Gemini, Qwen, Claude, DeepSeek, Codestral)       │
│  ↔ Unified Memory Graph (Neo4j) ↔ GAPO / Compression / Integrity │
└─────────────────────────────────────────────────────────────────┘
```

- **text_analytics**: Embeddings (sentence-transformers), NER & topics (spaCy).
- **neo4j_memory**: Graph storage, two-tier retrieval, Cytoscape/GEXF export.
- **gapo_optimizer**: Multi-factor node scoring (similarity, PageRank, entropy).
- **llm_manager**: Retrieve → summarize → prompt → LLM → add turn to graph.
- **persona_fusion**: Per-persona system prompts and model routing; shared graph.
- **memory_compression**: Clustering + summarization; **integrity_analyzer**: response checks.

---

## Requirements

- Python 3.10+
- Neo4j 4.x/5.x (bolt)
- See `requirements.txt`: FastAPI, neo4j, sentence-transformers, spacy, scikit-learn, networkx, scipy, python-louvain, etc.

---

## Setup

1. **Neo4j**: Run Neo4j (e.g. Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j`). Create a user and set:

   ```bash
   set NEO4J_URI=bolt://localhost:7687
   set NEO4J_USER=neo4j
   set NEO4J_PASSWORD=your_password
   ```

2. **Python**:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Config**: Optionally set Aikipedia (or other LLM) endpoint/model in `config.py` or env.

4. **Run API**:

   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```

- **UI**: http://127.0.0.1:8000/ui  
- **Docs**: http://127.0.0.1:8000/docs  
- **Health**: http://127.0.0.1:8000/health  
- **Metrics**: http://127.0.0.1:8000/metrics  

---

## Main API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness + Neo4j connectivity |
| GET | `/metrics` | Node/edge/session counts |
| POST | `/ask_json` | Query with memory retrieval and LLM response |
| POST | `/gapo_query` | Query with GAPO-optimized node selection |
| POST | `/ask_persona` | Query as a specific persona (e.g. gemini, claude) |
| GET | `/graph` | Full graph (Cytoscape JSON) |
| GET | `/subgraph?node_ids=...` | Subgraph by node IDs |
| POST | `/evaluate` | Retrieval metrics (CRS, Precision@K, Recall@K, GTE, MIU, SCI) |
| POST | `/compress` | Run memory compression (clustering + summarization) |
| POST | `/check_integrity` | Check LLM answer vs. retrieved memory |
| GET | `/export_neo4j_cypher` | Cypher queries for Neo4j Browser |
| GET | `/export_gephi` | GEXF export for Gephi |

---

## Project structure

- `app.py` — FastAPI app, routes, health/metrics.
- `config.py` — Neo4j, LLM, similarity threshold.
- `text_analytics.py` — Embeddings, NER, topics.
- `graph_memory.py` — In-memory graph (NetworkX) reference.
- `neo4j_memory.py` — Neo4j graph memory implementation.
- `gapo_optimizer.py` — GAPO scoring and node selection.
- `llm_manager.py` — LLM calls and memory summarization.
- `persona_fusion.py` — Multi-persona routing and prompts.
- `memory_compression.py` — Clustering and summarization.
- `integrity_analyzer.py` — Response integrity checks.

---

## License

Use and modify as needed for portfolio and research.
