# neo4j_memory.py

import uuid
import time
import json
from typing import Dict, Any, List, Tuple, Optional
from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from text_analytics import TextAnalytics

class Neo4jGraphMemory:
    """
    Neo4j-based implementation of the Graph Memory Layer.
    Replaces NetworkX with a proper graph database for better scalability.
    """
    
    def __init__(self, analytics: TextAnalytics, 
                 uri: str = "bolt://127.0.0.1:7687",
                 user: str = "neo4j", 
                 password: str = "Vennela_1512",
                 similarity_threshold: float = 0.82):
        self.analytics = analytics
        self.similarity_threshold = similarity_threshold
        self.session_last_node = {}
        
        # Neo4j connection
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize constraints and indexes
        self._initialize_database()
        print("Neo4j Graph Memory layer initialized.")
    
    def _initialize_database(self):
        """Create constraints and indexes for optimal performance."""
        with self.driver.session() as session:
            try:
                # Create constraints (with proper error handling)
                try:
                    session.run("CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE")
                except Exception as e:
                    # Constraint might already exist, that's OK
                    pass
                
                try:
                    session.run("CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE")
                except Exception as e:
                    pass
                
                # Create indexes for better query performance
                try:
                    session.run("CREATE INDEX conversation_timestamp IF NOT EXISTS FOR (c:Conversation) ON (c.timestamp)")
                except Exception as e:
                    pass
                
                try:
                    session.run("CREATE INDEX conversation_session IF NOT EXISTS FOR (c:Conversation) ON (c.session_id)")
                except Exception as e:
                    pass
                
                try:
                    session.run("CREATE INDEX conversation_topics IF NOT EXISTS FOR (c:Conversation) ON (c.topics)")
                except Exception as e:
                    pass
                
                try:
                    session.run("CREATE INDEX conversation_entities IF NOT EXISTS FOR (c:Conversation) ON (c.entities)")
                except Exception as e:
                    pass
            except Exception as e:
                # If Neo4j version doesn't support IF NOT EXISTS, try without it
                print(f"Note: Some constraints/indexes may already exist: {e}")
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
    
    def add_turn(self, query: str, response: str, session_id: str, timestamp: Optional[float] = None) -> str:
        """
        Adds a new conversation turn (query + response) to the Neo4j graph.
        
        Args:
            query: User query text
            response: AI response text
            session_id: Session identifier
            timestamp: Optional timestamp (default: current time)
        """
        # 1. Analyze query and response
        query_data = self.analytics.analyze(query)
        response_data = self.analytics.analyze(response)
        
        # 2. Create a single node for the turn
        node_id = str(uuid.uuid4())
        turn_text = f"User: {query}\nAI: {response}"
        combined_embedding = self.analytics.analyze(turn_text)['embedding']
        
        # Use provided timestamp or current time
        if timestamp is None:
            timestamp = time.time()
        
        # 3. Store in Neo4j
        with self.driver.session() as session:
            # Create conversation node (avoid 'query' kwarg collision by using 'user_query' param name)
            session.run(
                """
                CREATE (c:Conversation {
                    id: $node_id,
                    text: $text,
                    query: $user_query,
                    response: $response,
                    embedding: $embedding,
                    entities: $entities,
                    topics: $topics,
                    timestamp: $timestamp,
                    session_id: $session_id
                })
                """,
                {
                    "node_id": node_id,
                    "text": turn_text,
                    "user_query": query,
                    "response": response,
                    "embedding": combined_embedding,
                    "entities": list(set(e['text'] for e in query_data['entities'] + response_data['entities'])),
                    "topics": list(dict.fromkeys((query_data.get('topics', []) + response_data.get('topics', []))))[:15],
                    "timestamp": timestamp,
                    "session_id": session_id,
                }
            )
            
            # Create or update session node
            session.run(
                """
                MERGE (s:Session {id: $session_id})
                SET s.last_conversation = $node_id, s.updated_at = $timestamp
                """,
                {"session_id": session_id, "node_id": node_id, "timestamp": timestamp}
            )
        
        # 4. Update edges
        self._update_edges(node_id, session_id)
        
        # 5. Update the last node for this session
        self.session_last_node[session_id] = node_id
        
        return node_id
    
    def _update_edges(self, new_node_id: str, session_id: str):
        """
        Connects the new node to existing nodes based on temporal, semantic, and entity similarity.
        """
        with self.driver.session() as session:
            # 1. Capture previous node in session (temporal link target)
            prev_node_id = None
            if session_id in self.session_last_node and self.session_last_node[session_id] != new_node_id:
                prev_node_id = self.session_last_node[session_id]

            # 2. Get all existing conversations for similarity comparison
            result = session.run("""
                MATCH (c:Conversation)
                WHERE c.id <> $new_id
                RETURN c.id as id, c.embedding as embedding, c.entities as entities, c.topics as topics
            """, new_id=new_node_id)
            
            existing_nodes = list(result)
            if not existing_nodes:
                return
            
            # 3. Get new node data
            new_node_result = session.run("""
                MATCH (c:Conversation {id: $new_id})
                RETURN c.embedding as embedding, c.entities as entities, c.topics as topics
            """, new_id=new_node_id)
            
            new_node_data = new_node_result.single()
            if not new_node_data:
                return
            
            new_embedding = new_node_data['embedding']
            new_entities = set(new_node_data['entities'])
            new_topics = set(new_node_data['topics'])
            
            # 4. Calculate similarities
            similarity_by_id = {}
            for existing in existing_nodes:
                other_id = existing['id']
                other_embedding = existing['embedding']
                other_entities = set(existing['entities'])
                other_topics = set(existing['topics'])
                
                # Semantic similarity
                similarity = cosine_similarity([new_embedding], [other_embedding])[0][0]
                similarity_by_id[other_id] = float(similarity)
                if similarity > self.similarity_threshold:
                    session.run(
                        """
                        MATCH (a:Conversation {id: $new_id}), (b:Conversation {id: $other_id})
                        CREATE (a)-[:SEMANTIC {weight: $weight}]->(b)
                        """,
                        {"new_id": new_node_id, "other_id": other_id, "weight": float(similarity)}
                    )
                
                # Entity link; weight equals similarity between conversations
                shared_entities = new_entities.intersection(other_entities)
                if shared_entities:
                    weight = float(similarity)
                    session.run(
                        """
                        MATCH (a:Conversation {id: $new_id}), (b:Conversation {id: $other_id})
                        CREATE (a)-[:ENTITY {weight: $weight, shared: $shared}]->(b)
                        """,
                        {"new_id": new_node_id, "other_id": other_id, "weight": weight, "shared": list(shared_entities)}
                    )

            # 5. Temporal Edge with similarity weight (if applicable)
            if prev_node_id is not None:
                temp_weight = similarity_by_id.get(prev_node_id)
                if temp_weight is None:
                    # Fallback if prev node wasn't in existing list for some reason
                    prev_emb_res = session.run(
                        "MATCH (c:Conversation {id: $id}) RETURN c.embedding as embedding",
                        {"id": prev_node_id}
                    ).single()
                    if prev_emb_res:
                        temp_weight = float(cosine_similarity([new_embedding], [prev_emb_res['embedding']])[0][0])
                    else:
                        temp_weight = 1.0
                session.run(
                    """
                    MATCH (a:Conversation {id: $new_id}), (b:Conversation {id: $prev_id})
                    CREATE (a)-[:TEMPORAL {weight: $weight}]->(b)
                    """,
                    {"new_id": new_node_id, "prev_id": prev_node_id, "weight": float(temp_weight)}
                )
    
    def retrieve_subgraph(self, query: str, k: int = None, min_similarity: float = 0.3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Two-tier efficient retrieval:
        1. First search recent nodes (fast, O(500))
        2. If no relevant results (similarity < 0.35), search entire graph (O(n))
        
        Returns ALL nodes above similarity threshold (no k limit).
        Uses efficient cosine similarity for O(n*d) complexity where d is embedding dimension (constant).
        
        Args:
            query: The query string
            k: Optional limit (if None, returns all nodes above min_similarity)
            min_similarity: Minimum similarity threshold (default 0.3)
        """
        query_embedding = self.analytics.analyze(query)['embedding']
        SIMILARITY_THRESHOLD = 0.35  # If max similarity below this, search full graph
        RECENT_NODES_LIMIT = 500  # Search recent nodes first for speed
        
        with self.driver.session() as session:
            # Get total count for logging
            count_result = session.run("MATCH (c:Conversation) WHERE c.embedding IS NOT NULL RETURN count(c) as count")
            total_count = count_result.single()['count'] or 0
            
            # TIER 1: Search recent nodes first (O(500) = constant time)
            print(f"[RETRIEVE] Tier 1: Searching {min(RECENT_NODES_LIMIT, total_count)} recent nodes...")
            recent_result = session.run(
                f"""
                MATCH (c:Conversation)
                WHERE c.embedding IS NOT NULL
                OPTIONAL MATCH (c)-[r]-()
                WITH c, count(r) as degree
                RETURN c.id as id, c.embedding as embedding, c.text as text, c.timestamp as timestamp, degree
                ORDER BY c.timestamp DESC
                LIMIT {RECENT_NODES_LIMIT}
                """
            )
            recent_nodes_data = list(recent_result)
            
            if not recent_nodes_data:
                return {"nodes": [], "edges": []}, {}
            
            # Compute similarities for recent nodes
            recent_node_ids = []
            recent_embeddings = []
            recent_degrees = []
            recent_texts = []
            recent_timestamps = []
            
            for n in recent_nodes_data:
                if n.get('embedding') is None:
                    continue
                recent_node_ids.append(n['id'])
                recent_embeddings.append(n['embedding'])
                recent_degrees.append(n.get('degree', 0) or 0)
                recent_texts.append(n.get('text') or '')
                recent_timestamps.append(n.get('timestamp') or 0)
            
            if not recent_embeddings:
                return {"nodes": [], "edges": []}, {}
            
            # Fast similarity computation O(n*d) where d is constant
            recent_similarities = cosine_similarity([query_embedding], recent_embeddings)[0]
            max_recent_similarity = float(recent_similarities.max()) if len(recent_similarities) > 0 else 0.0
            
            print(f"[RETRIEVE] Tier 1 max similarity: {max_recent_similarity:.3f}")
            
            # TIER 2: If recent nodes don't have good matches, search entire graph
            if max_recent_similarity < SIMILARITY_THRESHOLD and total_count > RECENT_NODES_LIMIT:
                print(f"[RETRIEVE] Tier 2: Low similarity ({max_recent_similarity:.3f} < {SIMILARITY_THRESHOLD}), searching entire graph ({total_count} nodes)...")
                
                full_result = session.run(
                    """
                    MATCH (c:Conversation)
                    WHERE c.embedding IS NOT NULL
                    OPTIONAL MATCH (c)-[r]-()
                    WITH c, count(r) as degree
                    RETURN c.id as id, c.embedding as embedding, c.text as text, c.timestamp as timestamp, degree
                    ORDER BY c.timestamp DESC
                    """
                )
                all_nodes_data = list(full_result)
                
                # Use full graph data
                node_ids = []
                embeddings = []
                degrees = []
                texts = []
                timestamps = []
                
                for n in all_nodes_data:
                    if n.get('embedding') is None:
                        continue
                    node_ids.append(n['id'])
                    embeddings.append(n['embedding'])
                    degrees.append(n.get('degree', 0) or 0)
                    texts.append(n.get('text') or '')
                    timestamps.append(n.get('timestamp') or 0)
                
                if not embeddings:
                    return {"nodes": [], "edges": []}, {}
                
                similarities = cosine_similarity([query_embedding], embeddings)[0]
                retrieval_method = "Two-tier: Full graph search (low similarity in recent)"
            else:
                # Use recent nodes data (fast path)
                node_ids = recent_node_ids
                embeddings = recent_embeddings
                degrees = recent_degrees
                texts = recent_texts
                timestamps = recent_timestamps
                similarities = recent_similarities
                retrieval_method = f"Two-tier: Recent nodes only (similarity {max_recent_similarity:.3f} >= {SIMILARITY_THRESHOLD})"

            # Compute combined scores (similarity + degree)
            entry_node_idx = int(similarities.argmax())
            entry_node_id = node_ids[entry_node_idx]

            # Normalize degrees for scoring
            max_deg = max(degrees) if degrees else 1
            norm_degrees = [(d / max_deg) if max_deg else 0 for d in degrees]
            # Score: 0.7 similarity + 0.3 degree (graph importance)
            scores = [0.7 * float(sim) + 0.3 * float(norm_deg) for sim, norm_deg in zip(similarities, norm_degrees)]

            # Rank nodes and filter by similarity threshold (no k limit - return all relevant)
            ranked = sorted(zip(node_ids, similarities, scores, texts, timestamps), key=lambda x: x[2], reverse=True)
            
            # Filter by minimum similarity threshold instead of top-k
            if k is None:
                # Return all nodes above similarity threshold
                relevant_nodes = [(nid, sim, score, txt, ts) for nid, sim, score, txt, ts in ranked if sim >= min_similarity]
                print(f"[RETRIEVE] Found {len(relevant_nodes)} nodes above similarity threshold {min_similarity}")
            else:
                # If k is specified, use it as a limit
                relevant_nodes = ranked[:k]
                print(f"[RETRIEVE] Selected top {len(relevant_nodes)} nodes (k={k})")
            
            top_nodes_ranked = relevant_nodes
            top_node_ids = [nid for nid, _, _, _, _ in top_nodes_ranked]
            
            print(f"[RETRIEVE] Selected {len(top_node_ids)} nodes with method: {retrieval_method}")

            # 3) Fetch edges among top nodes
            edges_res = session.run(
                """
                MATCH (a:Conversation)-[r]->(b:Conversation)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id as source, b.id as target, type(r) as relType, r.weight as weight
                """,
                {"ids": top_node_ids}
            )
            edges = [
                {
                    'source': rec['source'],
                    'target': rec['target'],
                    'type': rec['relType'],
                    'weight': rec.get('weight', 1.0)
                }
                for rec in edges_res
            ]

            subgraph_data = {
                "nodes": [
                    {
                        'id': nid,
                        'text': txt,
                        'timestamp': ts,
                        'similarity': float(sim),
                        'score': float(score)
                    } for nid, sim, score, txt, ts in top_nodes_ranked
                ],
                "edges": edges
            }

            max_sim = float(similarities.max()) if len(similarities) > 0 else 0.0
            explanation = {
                "entry_point": entry_node_id,
                "entry_reason": f"Highest semantic similarity ({max_sim:.3f}) to query.",
                "retrieval_method": retrieval_method,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "min_similarity": min_similarity,
                "nodes_searched": len(node_ids),
                "total_nodes_in_db": total_count,
                "nodes_returned": len(top_nodes_ranked),
                "top_nodes": [(nid, float(sim), float(score)) for nid, sim, score, _, _ in top_nodes_ranked]
            }

            return subgraph_data, explanation
    
    def _generate_short_label(self, node: Dict[str, Any]) -> str:
        """
        Generate a short label from topics/keywords or use a shortened ID.
        """
        topics = node.get('topics', [])
        entities = node.get('entities', [])
        
        # Try to use first few topics as label
        if topics and len(topics) > 0:
            # Take first 2-3 topics, max 30 chars
            label_parts = []
            total_len = 0
            for topic in topics[:3]:
                if total_len + len(topic) + 2 <= 30:  # +2 for ", "
                    label_parts.append(topic)
                    total_len += len(topic) + 2
                else:
                    break
            if label_parts:
                return ", ".join(label_parts)
        
        # Fallback to first entity if available
        if entities and len(entities) > 0:
            entity_text = entities[0] if isinstance(entities[0], str) else entities[0].get('text', '')
            if entity_text and len(entity_text) <= 30:
                return entity_text
        
        # Final fallback: use short ID (first 8 chars)
        node_id = node.get('id', '')
        return node_id[:8] if node_id else "Conv"
    
    def nodes_by_topic(self, topic_query: str, k: int = 10) -> List[str]:
        """
        Return up to k node_ids whose topics include the topic_query.
        """
        topic_query_l = topic_query.lower()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Conversation)
                WHERE ANY(topic IN c.topics WHERE toLower(topic) CONTAINS $topic_query)
                RETURN c.id as id
                LIMIT $k
            """, topic_query=topic_query_l, k=k)
            
            return [record['id'] for record in result]
    
    def to_cytoscape(self, highlight_nodes: List[str] = None, node_ids: List[str] = None) -> Dict[str, Any]:
        """
        Export graph to Cytoscape JSON format for visualization.
        
        Args:
            highlight_nodes: List of node IDs to highlight
            node_ids: If provided, only return these nodes (and edges between them). 
                     If None, returns all nodes.
        """
        highlight = set(highlight_nodes or [])
        
        with self.driver.session() as session:
            if node_ids:
                # Return only specified nodes (subgraph)
                nodes_result = session.run("""
                    MATCH (c:Conversation)
                    WHERE c.id IN $node_ids
                    RETURN c.id as id, c.text as text, c.query as query, c.response as response,
                           c.timestamp as timestamp, c.session_id as session_id, 
                           c.topics as topics, c.entities as entities
                    ORDER BY c.timestamp DESC
                """, node_ids=node_ids)
                
                # Get edges only between specified nodes
                edges_result = session.run("""
                    MATCH (a:Conversation)-[r]->(b:Conversation)
                    WHERE a.id IN $node_ids AND b.id IN $node_ids
                    RETURN a.id as source, b.id as target, type(r) as relType, r.weight as weight
                """, node_ids=node_ids)
            else:
                # Get all nodes - optimized query
                nodes_result = session.run("""
                    MATCH (c:Conversation)
                    RETURN c.id as id, c.text as text, c.query as query, c.response as response,
                           c.timestamp as timestamp, c.session_id as session_id, 
                           c.topics as topics, c.entities as entities
                    ORDER BY c.timestamp DESC
                """)
                
                # Get all edges; use type(r) for relationship type - optimized
                edges_result = session.run("""
                    MATCH (a:Conversation)-[r]->(b:Conversation)
                    RETURN a.id as source, b.id as target, type(r) as relType, r.weight as weight
                """)
            
            elements = {
                "nodes": [
                    {
                        "data": {
                            "id": node['id'],
                            "label": self._generate_short_label(node),
                            "full_text": node['text'],  # Store full text for display on click
                            "query": node.get('query', ''),
                            "response": node.get('response', ''),
                            "type": "turn",
                            "session_id": node['session_id'],
                            "timestamp": node['timestamp'],
                            "topics": node['topics'],
                            "entities": node['entities'],
                            "highlight": node['id'] in highlight,
                        }
                    }
                    for node in nodes_result
                ],
                "edges": [
                    {
                        "data": {
                            "id": f"{edge['source']}-{edge['target']}",
                            "source": edge['source'],
                            "target": edge['target'],
                            "type": edge['relType'],
                            "weight": edge['weight']
                        }
                    }
                    for edge in edges_result
                ]
            }
            
            return elements
    
    def save(self, path: str) -> None:
        """
        Export graph data to JSON file (for backup/portability).
        """
        try:
            elements = self.to_cytoscape()
            data = {
                "graph": elements,
                "session_last_node": self.session_last_node,
                "similarity_threshold": self.similarity_threshold,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Failed to save graph memory: {e}")
    
    def load(self, path: str) -> None:
        """
        Load graph data from JSON file (for backup/portability).
        Note: This is a simplified import - full Neo4j restoration would need more complex logic.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # This is a basic implementation - in production you'd want more sophisticated import
            print("Note: Neo4j import from JSON is simplified. Consider using Neo4j's native import tools.")
            
            self.session_last_node = data.get("session_last_node", {})
            self.similarity_threshold = data.get("similarity_threshold", self.similarity_threshold)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Failed to load graph memory: {e}")
    
    def evaluate_retrieval(self, query: str, k: int, ground_truth_node_ids: List[str]) -> Dict[str, Any]:
        """
        Compute automated metrics for a single query using Neo4j.
        """
        import time as _time
        
        t0 = _time.time()
        subgraph_data, explanation = self.retrieve_subgraph(query, k=k)
        t1 = _time.time()
        
        retrieved_ids = [n['id'] for n in subgraph_data.get('nodes', [])]
        
        # CRS using analytics
        texts = [n['text'] for n in subgraph_data.get('nodes', [])]
        crs = self.analytics.contextual_relevance_score(query, texts) if texts else 0.0
        
        # Precision/Recall@K
        gt = set(ground_truth_node_ids)
        hit = [nid for nid in retrieved_ids if nid in gt]
        precision = (len(hit) / len(retrieved_ids)) if retrieved_ids else 0.0
        recall = (len(hit) / len(gt)) if gt else 0.0
        
        # GTE: nodes scored per ms
        elapsed_ms = max((t1 - t0) * 1000.0, 1.0)
        
        with self.driver.session() as session:
            total_nodes = session.run("MATCH (c:Conversation) RETURN count(c) as count").single()['count']
        
        gte = total_nodes / elapsed_ms
        
        # MIU: percent retrieved nodes with connections
        if retrieved_ids:
            with self.driver.session() as session:
                connected_count = session.run("""
                    MATCH (c:Conversation)
                    WHERE c.id IN $retrieved_ids
                    AND (c)-[]-()
                    RETURN count(c) as count
                """, retrieved_ids=retrieved_ids).single()['count']
            miu = connected_count / len(retrieved_ids)
        else:
            miu = 0.0
        
        # SCI: if top retrieved node shares session with most recent node
        sci = 0.0
        if self.session_last_node and retrieved_ids:
            with self.driver.session() as session:
                last_session = session.run("""
                    MATCH (c:Conversation {id: $last_id})
                    RETURN c.session_id as session_id
                """, last_id=list(self.session_last_node.values())[-1]).single()
                
                top_session = session.run("""
                    MATCH (c:Conversation {id: $top_id})
                    RETURN c.session_id as session_id
                """, top_id=retrieved_ids[0]).single()
                
                if last_session and top_session:
                    sci = 1.0 if last_session['session_id'] == top_session['session_id'] else 0.0
        
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
