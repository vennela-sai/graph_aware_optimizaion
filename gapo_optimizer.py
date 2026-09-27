# gapo_optimizer.py
"""
Graph-Aware Prompt Optimization (GAPO)
Mathematically optimizes which nodes to inject using similarity, PageRank, and informativeness.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import Counter
import math

# Try to import scipy entropy, fallback to manual calculation
try:
    from scipy.stats import entropy as scipy_entropy
    def entropy(probabilities, base=2):
        return float(scipy_entropy(probabilities, base=base))
except ImportError:
    def entropy(probabilities, base=2):
        return -sum(p * math.log(p, base) for p in probabilities if p > 0)

from text_analytics import TextAnalytics
from neo4j_memory import Neo4jGraphMemory


class GAPOOptimizer:
    """
    Graph-Aware Prompt Optimization using multi-factor scoring.
    """
    
    def __init__(self, memory: Neo4jGraphMemory, analytics: TextAnalytics):
        self.memory = memory
        self.analytics = analytics
        # Default weights: can be tuned
        self.alpha = 0.5  # similarity weight
        self.beta = 0.3   # PageRank weight
        self.gamma = 0.2  # informativeness weight
    
    def optimize_node_selection(self, query: str, k: int = 5, 
                                alpha: float = None, beta: float = None, gamma: float = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize node selection using GAPO formula:
        Score(n) = α·similarity(n,q) + β·pagerank(n) + γ·informativeness(n)
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        
        # Normalize weights
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        query_embedding = self.analytics.analyze(query)['embedding']
        
        with self.memory.driver.session() as session:
            # Get all nodes with embeddings
            result = session.run("""
                MATCH (c:Conversation)
                WHERE c.embedding IS NOT NULL
                OPTIONAL MATCH (c)-[r]-()
                WITH c, count(r) as degree
                RETURN c.id as id, c.embedding as embedding, c.text as text, 
                       c.timestamp as timestamp, degree
            """)
            
            nodes_data = list(result)
            if not nodes_data:
                return {"nodes": [], "edges": []}, {}
            
            # Calculate scores for each node
            node_scores = []
            similarities = []
            pageranks = []
            informativeness_scores = []
            
            for node in nodes_data:
                node_id = node['id']
                embedding = node['embedding']
                text = node.get('text', '')
                degree = node.get('degree', 0) or 0
                
                # 1. Similarity score
                similarity = float(np.dot(query_embedding, embedding) / 
                                 (np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8))
                similarities.append(similarity)
                
                # 2. PageRank (using degree as proxy, or calculate actual PageRank)
                # For now, use normalized degree
                pagerank = self._calculate_pagerank_proxy(node_id, nodes_data, session)
                pageranks.append(pagerank)
                
                # 3. Informativeness (Shannon entropy)
                informativeness = self._calculate_informativeness(text)
                informativeness_scores.append(informativeness)
            
            # Normalize scores
            if similarities:
                max_sim = max(similarities) or 1
                similarities = [s / max_sim for s in similarities]
            
            if pageranks:
                max_pr = max(pageranks) or 1
                pageranks = [pr / max_pr for pr in pageranks]
            
            if informativeness_scores:
                max_info = max(informativeness_scores) or 1
                informativeness_scores = [info / max_info for info in informativeness_scores]
            
            # Calculate combined scores
            combined_scores = []
            for i in range(len(nodes_data)):
                score = (self.alpha * similarities[i] + 
                        self.beta * pageranks[i] + 
                        self.gamma * informativeness_scores[i])
                combined_scores.append((nodes_data[i]['id'], score, 
                                       similarities[i], pageranks[i], informativeness_scores[i]))
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top K nodes
            top_k = combined_scores[:k]
            top_node_ids = [node_id for node_id, _, _, _, _ in top_k]
            
            # Get edges between top nodes
            edges_result = session.run("""
                MATCH (a:Conversation)-[r]->(b:Conversation)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id as source, b.id as target, type(r) as relType, r.weight as weight
            """, ids=top_node_ids)
            
            edges = [
                {
                    'source': rec['source'],
                    'target': rec['target'],
                    'type': rec['relType'],
                    'weight': rec.get('weight', 1.0)
                }
                for rec in edges_result
            ]
            
            # Get full node data
            nodes_result = session.run("""
                MATCH (c:Conversation)
                WHERE c.id IN $ids
                RETURN c.id as id, c.text as text, c.timestamp as timestamp,
                       c.query as query, c.response as response
            """, ids=top_node_ids)
            
            nodes = []
            score_map = {node_id: {'sim': sim, 'pr': pr, 'info': info, 'combined': combined} 
                        for node_id, combined, sim, pr, info in top_k}
            
            for record in nodes_result:
                node_id = record['id']
                scores = score_map.get(node_id, {'sim': 0, 'pr': 0, 'info': 0, 'combined': 0})
                nodes.append({
                    'id': node_id,
                    'text': record.get('text', ''),
                    'timestamp': record.get('timestamp', 0),
                    'query': record.get('query', ''),
                    'response': record.get('response', '')
                })
            
            subgraph_data = {"nodes": nodes, "edges": edges}
            
            explanation = {
                "method": "GAPO (Graph-Aware Prompt Optimization)",
                "formula": f"Score = {self.alpha:.2f}·similarity + {self.beta:.2f}·pagerank + {self.gamma:.2f}·informativeness",
                "top_k_nodes": [
                    {
                        "node_id": node_id,
                        "combined": combined,
                        "combined_score": combined,
                        "similarity": sim,
                        "pagerank": pr,
                        "informativeness": info
                    }
                    for node_id, combined, sim, pr, info in top_k
                ]
            }
            
            return subgraph_data, explanation
    
    def _calculate_pagerank_proxy(self, node_id: str, all_nodes: List, session) -> float:
        """
        Calculate PageRank proxy using degree and connections.
        For full PageRank, would need to build full graph and run algorithm.
        """
        # Get degree (number of connections)
        node_data = next((n for n in all_nodes if n['id'] == node_id), None)
        if not node_data:
            return 0.0
        
        degree = node_data.get('degree', 0) or 0
        
        # Get weighted degree (sum of edge weights)
        result = session.run("""
            MATCH (c:Conversation {id: $node_id})-[r]-()
            RETURN sum(r.weight) as weighted_degree
        """, node_id=node_id)
        
        weighted_degree = result.single()['weighted_degree'] or 0.0
        
        # Combine degree and weighted degree
        return float(degree * 0.5 + weighted_degree * 0.5)
    
    def _calculate_informativeness(self, text: str) -> float:
        """
        Calculate Shannon entropy (informativeness) of text.
        Higher entropy = more informative.
        """
        if not text:
            return 0.0
        
        # Tokenize (simple word-based)
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        # Calculate word frequency
        word_counts = Counter(words)
        total_words = len(words)
        
        # Calculate probabilities
        probabilities = [count / total_words for count in word_counts.values()]
        
        # Calculate Shannon entropy
        entropy_value = entropy(probabilities, base=2)
        
        # Normalize by log2 of vocabulary size (max entropy)
        max_entropy = math.log2(len(word_counts)) if word_counts else 1
        normalized_entropy = entropy_value / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)

