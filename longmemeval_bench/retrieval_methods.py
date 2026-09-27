"""Session-level retrieval: BM25, dense MiniLM, two-tier, GAPO."""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def entropy_text(text: str) -> float:
    words = text.lower().split()
    if len(words) <= 1:
        return 0.0
    c = Counter(words)
    n = len(words)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in c.values())


def bm25_rank(query: str, corpus: List[str]) -> List[int]:
    tokenized = [d.split() for d in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    return list(np.argsort(scores)[::-1])


def dense_rank(
    model: SentenceTransformer, query: str, corpus: List[str]
) -> List[int]:
    q_emb = model.encode(query, convert_to_numpy=True)
    doc_emb = model.encode(corpus, convert_to_numpy=True)
    scores = (doc_emb @ q_emb) / (
        np.linalg.norm(doc_emb, axis=1) * np.linalg.norm(q_emb) + 1e-8
    )
    return list(np.argsort(scores)[::-1])


def two_tier_session_rank(
    model: SentenceTransformer,
    query: str,
    corpus: List[str],
    corpus_timestamps: List[str],
    L: int = 20,
    rho_min: float = 0.35,
    tau_min: float = 0.3,
    lam: float = 0.7,
) -> List[int]:
    """Tier-1: last L sessions by time; Tier-2: full pool if max sim in Tier-1 < rho_min. Rank by lam*sim + (1-lam)*norm_deg."""
    n = len(corpus)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: corpus_timestamps[i])
    recent = order[-min(L, n) :]
    q_emb = model.encode(query, convert_to_numpy=True)
    embs = model.encode(corpus, convert_to_numpy=True)
    sims = (embs @ q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    rho = float(np.max(sims[recent])) if recent else 0.0
    if rho < rho_min and n > L:
        cand = list(range(n))
    else:
        cand = list(recent)

    embs_c = embs[cand]
    sims_c = sims[cand]
    deg = np.zeros(len(cand))
    for ii in range(len(cand)):
        for jj in range(len(cand)):
            if ii != jj:
                deg[ii] += max(0.0, cosine_sim(embs_c[ii], embs_c[jj]) - 0.5)
    dmax = float(deg.max()) if deg.max() > 0 else 1.0
    norm_deg = deg / dmax
    score = lam * sims_c + (1 - lam) * norm_deg
    valid = sims_c >= tau_min
    if not np.any(valid):
        valid = np.ones_like(valid, dtype=bool)
    order_c = np.argsort(np.where(valid, score, -1e9))[::-1]
    ranked = [cand[k] for k in order_c]
    for i in range(n):
        if i not in ranked:
            ranked.append(i)
    return ranked


def gapo_session_rank(
    model: SentenceTransformer,
    query: str,
    corpus: List[str],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    theta_edge: float = 0.35,
) -> List[int]:
    """GAPO-style ranking on session documents (graph approximated on candidate pool)."""
    n = len(corpus)
    if n == 0:
        return []
    q_emb = model.encode(query, convert_to_numpy=True)
    doc_emb = model.encode(corpus, convert_to_numpy=True)
    sims = (doc_emb @ q_emb) / (
        np.linalg.norm(doc_emb, axis=1) * np.linalg.norm(q_emb) + 1e-8
    )
    max_sim = sims.max() + 1e-8
    s_tilde = sims / max_sim

    deg = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                c = cosine_sim(doc_emb[i], doc_emb[j])
                if c > theta_edge:
                    deg[i] += c
    max_deg = deg.max() + 1e-8
    p_tilde = deg / max_deg

    h_raw = np.array([entropy_text(corpus[i]) for i in range(n)])
    max_h = h_raw.max() + 1e-8
    h_tilde = h_raw / max_h

    total = alpha + beta + gamma
    a, b, g = alpha / total, beta / total, gamma / total
    gapo = a * s_tilde + b * p_tilde + g * h_tilde
    return list(np.argsort(gapo)[::-1])
