# text_analytics.py

import spacy
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity

class TextAnalytics:
    """
    A class to handle text processing: embedding generation and entity extraction.
    """
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.nlp = spacy.load("en_core_web_sm")
        print("Text Analytics layer initialized.")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Processes a text string to extract embeddings, entities, and other features.
        """
        doc = self.nlp(text)
        
        # 1. Extract named entities
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        # 1b. Extract lightweight topics via noun chunks and frequent lemmas
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        candidate_tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop and token.pos_ in {"NOUN", "PROPN"}
        ]
        # Deduplicate while preserving order
        seen = set()
        topics: List[str] = []
        for phrase in noun_chunks + candidate_tokens:
            if phrase not in seen:
                seen.add(phrase)
                topics.append(phrase)
        topics = topics[:15]

        # 2. Generate sentence embedding
        embedding = self.embedding_model.encode(text, convert_to_tensor=False).tolist()
        
        return {
            "text": text,
            "embedding": embedding,
            "entities": entities,
            "topics": topics,
        }

    # -------------------- Metrics helpers --------------------
    def contextual_relevance_score(self, query: str, docs: List[str]) -> float:
        """
        Average cosine similarity between query embedding and doc embeddings.
        """
        if not docs:
            return 0.0
        q_emb = self.embedding_model.encode(query, convert_to_tensor=False)
        d_embs = self.embedding_model.encode(docs, convert_to_tensor=False)
        sims = cosine_similarity([q_emb], d_embs)[0]
        return float(sum(sims) / len(sims))

