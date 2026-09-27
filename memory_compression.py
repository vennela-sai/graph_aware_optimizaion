# memory_compression.py
"""
Dynamic Memory Compression: Neural + Graph Hybrid Summarizer
Uses graph clustering to compress large graphs while maintaining semantic meaning.
"""

import networkx as nx
from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict
import time

from text_analytics import TextAnalytics
from neo4j_memory import Neo4jGraphMemory


class MemoryCompressor:
    """
    Compresses large graphs using Louvain clustering and neural summarization.
    """
    
    def __init__(self, memory: Neo4jGraphMemory, analytics: TextAnalytics, compression_threshold: int = 10000):
        self.memory = memory
        self.analytics = analytics
        self.compression_threshold = compression_threshold
    
    def should_compress(self) -> bool:
        """Check if graph is large enough to warrant compression."""
        with self.memory.driver.session() as session:
            result = session.run("MATCH (c:Conversation) RETURN count(c) as count")
            count = result.single()['count']
            return count >= self.compression_threshold
    
    def get_clusters(self, k: int = 10) -> Dict[int, List[str]]:
        """
        Use Louvain-like clustering on the graph.
        Returns clusters as {cluster_id: [node_ids]}
        """
        # Build NetworkX graph from Neo4j for clustering
        with self.memory.driver.session() as session:
            # Get all nodes and edges
            nodes_result = session.run("MATCH (c:Conversation) RETURN c.id as id")
            node_ids = [record['id'] for record in nodes_result]
            
            edges_result = session.run("""
                MATCH (a:Conversation)-[r]->(b:Conversation)
                RETURN a.id as source, b.id as target, r.weight as weight
            """)
            
            G = nx.Graph()
            G.add_nodes_from(node_ids)
            
            for edge in edges_result:
                G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
            
            # Use community detection (Louvain algorithm)
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G)
            except ImportError:
                # Fallback: use simple connected components or label propagation
                try:
                    communities_generator = nx.community.label_propagation_communities(G)
                    communities = {}
                    for i, community in enumerate(communities_generator):
                        for node in community:
                            communities[node] = i
                except:
                    # Final fallback: use connected components
                    communities = {}
                    for i, component in enumerate(nx.connected_components(G)):
                        for node in component:
                            communities[node] = i
            
            # Group nodes by cluster
            clusters = defaultdict(list)
            for node_id, cluster_id in communities.items():
                clusters[cluster_id].append(node_id)
            
            # Filter to keep only clusters with multiple nodes
            significant_clusters = {cid: nodes for cid, nodes in clusters.items() if len(nodes) >= 3}
            
            return significant_clusters
    
    def summarize_cluster(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Generate a neural summary for a cluster of nodes.
        """
        with self.memory.driver.session() as session:
            result = session.run("""
                MATCH (c:Conversation)
                WHERE c.id IN $node_ids
                RETURN c.text as text, c.timestamp as timestamp
                ORDER BY c.timestamp
            """, node_ids=node_ids)
            
            texts = [record['text'] for record in result]
            if not texts:
                return None
            
            # Combine all texts
            combined_text = "\n\n".join(texts)
            
            # Use LLM to summarize (via memory's summarization)
            from llm_manager import LLMManager
            manager = LLMManager(self.memory)
            
            subgraph_data = {"nodes": [{"id": nid, "text": txt, "timestamp": 0} for nid, txt in zip(node_ids, texts)]}
            summary = manager._summarize_subgraph(subgraph_data)
            
            # Get average embedding
            embeddings = [self.analytics.analyze(text)['embedding'] for text in texts]
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            
            # Extract topics and entities from all nodes
            all_topics = set()
            all_entities = set()
            for text in texts:
                analysis = self.analytics.analyze(text)
                all_topics.update(analysis.get('topics', []))
                all_entities.update([e['text'] for e in analysis.get('entities', [])])
            
            return {
                "summary": summary,
                "embedding": avg_embedding,
                "topics": list(all_topics)[:15],
                "entities": list(all_entities),
                "compressed_from": node_ids,
                "node_count": len(node_ids)
            }
    
    def compress_graph(self, min_cluster_size: int = 3) -> Dict[str, Any]:
        """
        Main compression function.
        Returns statistics about compression.
        """
        if not self.should_compress():
            return {"status": "not_needed", "message": "Graph size below compression threshold"}
        
        print("Starting graph compression...")
        clusters = self.get_clusters()
        
        if not clusters:
            return {"status": "no_clusters", "message": "No significant clusters found"}
        
        compressed_count = 0
        total_nodes_compressed = 0
        
        with self.memory.driver.session() as session:
            for cluster_id, node_ids in clusters.items():
                if len(node_ids) < min_cluster_size:
                    continue
                
                # Generate summary
                summary_data = self.summarize_cluster(node_ids)
                if not summary_data:
                    continue
                
                # Create compressed node
                compressed_node_id = f"compressed_{cluster_id}_{int(time.time())}"
                
                session.run("""
                    CREATE (c:Conversation:Compressed {
                        id: $node_id,
                        text: $text,
                        query: $summary,
                        response: $summary,
                        embedding: $embedding,
                        entities: $entities,
                        topics: $topics,
                        timestamp: $timestamp,
                        session_id: 'compressed',
                        compressed_from: $compressed_from,
                        compression_date: $timestamp
                    })
                """, {
                    "node_id": compressed_node_id,
                    "text": f"Compressed Summary: {summary_data['summary']}",
                    "summary": summary_data['summary'],
                    "embedding": summary_data['embedding'],
                    "entities": summary_data['entities'],
                    "topics": summary_data['topics'],
                    "timestamp": time.time(),
                    "compressed_from": summary_data['compressed_from']
                })
                
                # Rewire edges: connect compressed node to nodes outside cluster
                # Get edges from original nodes to external nodes
                external_edges = session.run("""
                    MATCH (a:Conversation)-[r]->(b:Conversation)
                    WHERE a.id IN $cluster_ids AND NOT b.id IN $cluster_ids
                    RETURN DISTINCT b.id as target, AVG(r.weight) as avg_weight
                """, cluster_ids=node_ids)
                
                for edge in external_edges:
                    session.run("""
                        MATCH (comp:Conversation {id: $comp_id}), (target:Conversation {id: $target_id})
                        CREATE (comp)-[:COMPRESSED_EDGE {weight: $weight, type: 'SEMANTIC'}]->(target)
                    """, {
                        "comp_id": compressed_node_id,
                        "target_id": edge['target'],
                        "weight": edge['avg_weight']
                    })
                
                # Optionally delete original nodes (or mark as compressed)
                # For safety, we'll mark them instead of deleting
                session.run("""
                    MATCH (c:Conversation)
                    WHERE c.id IN $node_ids
                    SET c.compressed = true, c.compressed_into = $compressed_id
                """, {
                    "node_ids": node_ids,
                    "compressed_id": compressed_node_id
                })
                
                compressed_count += 1
                total_nodes_compressed += len(node_ids)
        
        return {
            "status": "success",
            "clusters_compressed": compressed_count,
            "nodes_compressed": total_nodes_compressed,
            "compression_ratio": f"{(total_nodes_compressed / (total_nodes_compressed + compressed_count) * 100):.1f}%"
        }

