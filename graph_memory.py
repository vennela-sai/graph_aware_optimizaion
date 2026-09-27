# graph_memory.py

import networkx as nx
import uuid
import time
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List, Tuple
import json

from text_analytics import TextAnalytics

class GraphMemory:
    """
    Manages the conversational memory as a graph using NetworkX.
    """
    def __init__(self, analytics: TextAnalytics, similarity_threshold: float = 0.82):
        self.graph = nx.Graph()
        self.analytics = analytics
        self.similarity_threshold = similarity_threshold
        self.session_last_node = {} # Maps session_id to the last node_id
        print("Graph Memory layer initialized.")

    def add_turn(self, query: str, response: str, session_id: str) -> str:
        """
        Adds a new conversation turn (query + response) to the graph.
        """
        # 1. Analyze query and response
        query_data = self.analytics.analyze(query)
        response_data = self.analytics.analyze(response)
        
        # 2. Create a single node for the turn
        node_id = str(uuid.uuid4())
        turn_text = f"User: {query}\nAI: {response}"
        combined_embedding = self.analytics.analyze(turn_text)['embedding']

        self.graph.add_node(
            node_id,
            type="turn",
            text=turn_text,
            query=query,
            response=response,
            embedding=combined_embedding,
            entities=list(set(e['text'] for e in query_data['entities'] + response_data['entities'])),
            topics=list(dict.fromkeys((query_data.get('topics', []) + response_data.get('topics', []))))[:15],
            timestamp=time.time(),
            session_id=session_id
        )
        
        # 3. Update edges to connect this node to the graph
        self._update_edges(node_id, session_id)
        
        # 4. Update the last node for this session
        self.session_last_node[session_id] = node_id
        
        return node_id

    def _update_edges(self, new_node_id: str, session_id: str):
        """
        Connects the new node to existing nodes based on temporal, semantic, and entity similarity.
        """
        new_node_data = self.graph.nodes[new_node_id]
        
        # 1. Temporal Edge (within the same session)
        if session_id in self.session_last_node and self.session_last_node[session_id] != new_node_id:
            prev_node_id = self.session_last_node[session_id]
            self.graph.add_edge(new_node_id, prev_node_id, type="temporal", weight=1.0)

        # 2. Semantic and Entity Edges (across all sessions)
        all_embeddings = [data['embedding'] for _, data in self.graph.nodes(data=True)]
        node_ids = list(self.graph.nodes())
        
        if len(all_embeddings) > 1:
            similarities = cosine_similarity([new_node_data['embedding']], all_embeddings)[0]
            
            for i, score in enumerate(similarities):
                if node_ids[i] == new_node_id:
                    continue
                
                # Semantic Edge
                if score > self.similarity_threshold:
                    self.graph.add_edge(new_node_id, node_ids[i], type="semantic", weight=float(score))
                
                # Entity Edge
                other_node_data = self.graph.nodes[node_ids[i]]
                shared_entities = set(new_node_data['entities']).intersection(set(other_node_data.get('entities', [])))
                if shared_entities:
                    self.graph.add_edge(new_node_id, node_ids[i], type="entity", weight=0.5 * len(shared_entities), shared=list(shared_entities))


    def retrieve_subgraph(self, query: str, k: int = 5) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Finds the most relevant subgraph for a new query.
        """
        if not self.graph.nodes:
            return nx.Graph(), {}

        query_embedding = self.analytics.analyze(query)['embedding']
        
        # 1. Find the entry point: the most similar node to the query
        nodes_data = self.graph.nodes(data=True)
        node_ids = [n[0] for n in nodes_data]
        embeddings = [n[1]['embedding'] for n in nodes_data]
        
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        entry_node_idx = similarities.argmax()
        entry_node_id = node_ids[entry_node_idx]
        
        # 2. Traverse from the entry point (weighted BFS/Personalized PageRank concept)
        # Using PageRank to find important nodes around the entry point
        personalization = {n: 0 for n in self.graph.nodes()}
        personalization[entry_node_id] = 1.0 # Bias the random walk to start here
        
        scores = nx.pagerank(self.graph, alpha=0.85, personalization=personalization, weight='weight')
        
        # 3. Select top-K nodes
        sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_k_node_ids = [node_id for node_id, score in sorted_nodes[:k]]
        
        subgraph = self.graph.subgraph(top_k_node_ids)
        
        explanation = {
            "entry_point": entry_node_id,
            "entry_reason": f"Highest semantic similarity ({similarities.max():.2f}) to query.",
            "retrieval_method": "Personalized PageRank",
            "top_k_nodes": sorted_nodes[:k]
        }
        
        return subgraph, explanation

    # -------------------- Evaluation Metrics --------------------
    def evaluate_retrieval(self, query: str, k: int, ground_truth_node_ids: List[str]) -> Dict[str, Any]:
        """
        Compute automated metrics for a single query.
        - CRS: contextual relevance score between query and retrieved node texts
        - Recall@K, Precision@K
        - GTE: number of nodes scored per millisecond (approx)
        - MIU: fraction of retrieved nodes that contributed semantic edges in last add_turn
        - SCI: whether the top node shares the session with last session turn (0/1 proxy)
        """
        import time as _time

        t0 = _time.time()
        subgraph, explanation = self.retrieve_subgraph(query, k=k)
        t1 = _time.time()

        retrieved_ids = list(subgraph.nodes())
        # CRS using analytics
        texts = [self.graph.nodes[nid].get('text', '') for nid in retrieved_ids]
        crs = self.analytics.contextual_relevance_score(query, texts) if texts else 0.0

        # Precision/Recall@K
        gt = set(ground_truth_node_ids)
        hit = [nid for nid in retrieved_ids if nid in gt]
        precision = (len(hit) / len(retrieved_ids)) if retrieved_ids else 0.0
        recall = (len(hit) / len(gt)) if gt else 0.0

        # GTE: nodes scored per ms (approximate by graph size / latency)
        elapsed_ms = max((t1 - t0) * 1000.0, 1.0)
        gte = len(self.graph.nodes) / elapsed_ms

        # MIU: percent retrieved nodes with degree > 1 (as proxy for being connected semantically)
        if retrieved_ids:
            util = [1 for nid in retrieved_ids if self.graph.degree(nid) > 1]
            miu = sum(util) / len(retrieved_ids)
        else:
            miu = 0.0

        # SCI: if top retrieved node shares session with most recent node globally
        sci = 0.0
        if self.session_last_node:
            last_node_id = list(self.session_last_node.values())[-1]
            last_session = self.graph.nodes[last_node_id].get('session_id')
            if retrieved_ids:
                top_session = self.graph.nodes[retrieved_ids[0]].get('session_id')
                sci = 1.0 if top_session == last_session else 0.0

        return {
            "CRS": float(crs),
            "Precision@K": float(precision),
            "Recall@K": float(recall),
            "GTE": float(gte),
            "MIU": float(miu),
            "SCI": float(sci),
            "retrieved_node_ids": retrieved_ids,
            "explanation": explanation,
        }

    # -------------------- Persistence --------------------
    def save(self, path: str) -> None:
        """
        Persist the current graph and session state to disk as JSON.
        """
        try:
            data = {
                "nodes": [
                    {
                        "id": node_id,
                        **{k: v for k, v in attrs.items() if k != "embedding"},
                        "embedding": attrs.get("embedding", []),
                    }
                    for node_id, attrs in self.graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "u": u,
                        "v": v,
                        **d,
                    }
                    for u, v, d in self.graph.edges(data=True)
                ],
                "session_last_node": self.session_last_node,
                "similarity_threshold": self.similarity_threshold,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Failed to save graph memory: {e}")

    def load(self, path: str) -> None:
        """
        Load graph and session state from a JSON file.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph.clear()
            # Rebuild nodes
            for node in data.get("nodes", []):
                node_id = node.pop("id")
                self.graph.add_node(node_id, **node)
            # Rebuild edges
            for edge in data.get("edges", []):
                u = edge.pop("u")
                v = edge.pop("v")
                self.graph.add_edge(u, v, **edge)
            self.session_last_node = data.get("session_last_node", {})
            self.similarity_threshold = data.get("similarity_threshold", self.similarity_threshold)
        except FileNotFoundError:
            # Fresh start, nothing to load
            pass
        except Exception as e:
            print(f"Failed to load graph memory: {e}")

    # -------------------- Utilities for Retrieval & Viz --------------------
    def nodes_by_topic(self, topic_query: str, k: int = 10) -> List[str]:
        """
        Return up to k node_ids whose topics include the topic_query (case-insensitive substring).
        """
        topic_query_l = topic_query.lower()
        matches = []
        for node_id, data in self.graph.nodes(data=True):
            topics = [t.lower() for t in data.get('topics', [])]
            if any(topic_query_l in t for t in topics):
                matches.append(node_id)
            if len(matches) >= k:
                break
        return matches

    def to_cytoscape(self, highlight_nodes: List[str] = None) -> Dict[str, Any]:
        """
        Export graph to Cytoscape JSON format for simple frontend visualization.
        highlight_nodes will be marked for UI emphasis.
        """
        highlight = set(highlight_nodes or [])
        elements = {
            "nodes": [
                {
                    "data": {
                        "id": node_id,
                        "label": data.get("text", "")[:80],
                        "type": data.get("type", "turn"),
                        "session_id": data.get("session_id"),
                        "timestamp": data.get("timestamp"),
                        "topics": data.get("topics", []),
                        "entities": data.get("entities", []),
                        "highlight": node_id in highlight,
                    }
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "data": {
                        "id": f"{u}-{v}",
                        "source": u,
                        "target": v,
                        "type": d.get("type", "semantic"),
                        "weight": d.get("weight", 1.0),
                        "shared": d.get("shared", []),
                    }
                }
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        return elements