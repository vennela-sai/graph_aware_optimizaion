from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from typing import Dict, Any, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import os

from text_analytics import TextAnalytics
from neo4j_memory import Neo4jGraphMemory
from llm_manager import LLMManager
from config import Config
from memory_compression import MemoryCompressor
from integrity_analyzer import IntegrityAnalyzer
from persona_fusion import PersonaFusion
from gapo_optimizer import GAPOOptimizer

app = FastAPI(title="Graph Memory API")

# Enable CORS for local dev and generic clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    analytics = TextAnalytics()
    neo4j_config = Config.get_neo4j_config()
    memory = Neo4jGraphMemory(
        analytics=analytics,
        uri=neo4j_config["uri"],
        user=neo4j_config["user"],
        password=neo4j_config["password"],
        similarity_threshold=Config.SIMILARITY_THRESHOLD
    )
    manager = LLMManager(memory)
    compressor = MemoryCompressor(memory, analytics)
    integrity_checker = IntegrityAnalyzer(memory, analytics)
    persona_manager = PersonaFusion(memory, manager)
    gapo_optimizer = GAPOOptimizer(memory, analytics)
    print("[OK] All modules initialized successfully")
except Exception as e:
    print(f"[ERROR] Error initializing modules: {e}")
    import traceback
    traceback.print_exc()
    raise

@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoints": [
            "/ask (POST form)",
            "/ask_json (POST json)",
            "/graph (GET)",
            "/export (GET)",
            "/import (POST file)",
            "/summarize_topic (GET)",
            "/evaluate (POST json)",
            "/compress (POST json)",
            "/check_integrity (POST json)",
            "/ask_persona (POST json)",
            "/gapo_query (POST json)",
            "/docs (Swagger UI)",
        ],
    }

@app.get("/favicon.ico")
async def favicon():
    # Avoid 404 noise in logs
    return Response(status_code=204)

@app.get("/ui")
async def ui():
    return FileResponse("frontend.html")

@app.get("/agent/{persona}")
async def agent_page(persona: str):
    """
    Serve separate page for each AI agent.
    Each agent has its own interface but shares the same Neo4j database.
    """
    valid_personas = ["gemini", "qwen", "claude", "deepseek", "codestral"]
    if persona not in valid_personas:
        return JSONResponse({"error": f"Invalid persona. Valid: {valid_personas}"}, status_code=400)
    
    filename = f"agent_{persona}.html"
    if os.path.exists(filename):
        return FileResponse(filename)
    else:
        return JSONResponse({"error": f"Agent page not found. Run: python create_agent_pages.py"}, status_code=404)

# Optional: expose project directory as /static if you want to load other assets
app.mount("/static", StaticFiles(directory="."), name="static")

@app.post("/ask")
async def ask(query: str = Form(...), session_id: str = Form("default")):
    result = manager.ask(query, session_id=session_id)
    return JSONResponse(result)

@app.post("/ask_json")
async def ask_json(payload: Dict[str, Any]):
    query = payload.get("query", "")
    session_id = payload.get("session_id", "default")
    print(f"[API] /ask_json called: query='{query[:50]}...', session_id='{session_id}'")
    try:
        result = manager.ask(query, session_id=session_id)
        print(f"[API] Returning result with answer length: {len(result.get('answer', ''))}")
        return JSONResponse(result)
    except Exception as e:
        print(f"[API ERROR] Exception in ask_json: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "error": str(e), 
            "answer": f"Error: {str(e)}",
            "explanation": {},
            "subgraph_node_ids": []
        }, status_code=500)

@app.get("/graph")
async def graph(highlight_query: str | None = None):
    highlight = []
    if highlight_query:
        highlight = memory.nodes_by_topic(highlight_query)
    
    # Log graph stats for debugging
    with memory.driver.session() as session:
        count_result = session.run("MATCH (c:Conversation) RETURN count(c) as count")
        total_nodes = count_result.single()['count'] or 0
        print(f"[GRAPH] Returning graph with {total_nodes} total nodes in Neo4j")
    
    return JSONResponse(memory.to_cytoscape(highlight_nodes=highlight))

@app.get("/subgraph")
async def subgraph_get(node_ids: str):
    """
    Return only specific nodes (subgraph) for visualization.
    node_ids: Comma-separated list of node IDs
    """
    node_id_list = [nid.strip() for nid in node_ids.split(",") if nid.strip()]
    
    if not node_id_list:
        return JSONResponse({"error": "No node IDs provided"}, status_code=400)
    
    print(f"[SUBGRAPH] Returning {len(node_id_list)} nodes")
    return JSONResponse(memory.to_cytoscape(node_ids=node_id_list))

@app.post("/subgraph")
async def subgraph_post(payload: Dict[str, Any]):
    """
    Return only specific nodes (subgraph) for visualization.
    Accepts JSON: {"node_ids": ["id1", "id2", ...]}
    """
    node_ids = payload.get("node_ids", [])
    
    if not node_ids:
        return JSONResponse({"error": "No node IDs provided"}, status_code=400)
    
    print(f"[SUBGRAPH] Returning {len(node_ids)} nodes")
    return JSONResponse(memory.to_cytoscape(node_ids=node_ids))

@app.get("/debug/check_topic")
async def check_topic(topic: str):
    """
    Debug endpoint to check if conversations about a topic exist in Neo4j.
    """
    with memory.driver.session() as session:
        # Search for conversations containing the topic
        result = session.run(
            """
            MATCH (c:Conversation)
            WHERE toLower(c.text) CONTAINS toLower($topic)
               OR toLower(c.query) CONTAINS toLower($topic)
            RETURN c.id as id, c.text as text, c.timestamp as timestamp, c.session_id as session_id
            ORDER BY c.timestamp DESC
            LIMIT 10
            """,
            {"topic": topic}
        )
        matches = [dict(record) for record in result]
        
        # Get total count
        count_result = session.run(
            """
            MATCH (c:Conversation)
            WHERE toLower(c.text) CONTAINS toLower($topic)
               OR toLower(c.query) CONTAINS toLower($topic)
            RETURN count(c) as count
            """,
            {"topic": topic}
        )
        total_count = count_result.single()['count'] or 0
        
        return JSONResponse({
            "topic": topic,
            "total_matches": total_count,
            "sample_matches": matches,
            "message": f"Found {total_count} conversations containing '{topic}'"
        })

@app.get("/export")
async def export_graph():
    # Return a JSON snapshot for download
    elements = memory.to_cytoscape()
    return JSONResponse({"graph": elements})

@app.get("/export_neo4j_cypher")
async def export_neo4j_cypher():
    """
    Export graph as Neo4j Cypher queries for viewing in Neo4j Browser.
    The graph is already in Neo4j, so this just provides visualization queries.
    Fast - doesn't fetch all data, just generates queries.
    """
    try:
        # Get counts quickly (these are fast queries)
        with memory.driver.session() as session:
            count_result = session.run("MATCH (c:Conversation) RETURN count(c) as count")
            total_nodes = count_result.single()['count'] or 0
            
            edge_count_result = session.run("MATCH (a:Conversation)-[r]->(b:Conversation) RETURN count(r) as count")
            total_edges = edge_count_result.single()['count'] or 0
    except Exception as e:
        print(f"[EXPORT] Error getting counts: {e}")
        total_nodes = 0
        total_edges = 0
    
    # Generate queries (no data fetching, just text)
    cypher_queries = []
    cypher_queries.append("// ========================================")
    cypher_queries.append("// Graph Memory - Neo4j Browser Visualization")
    cypher_queries.append(f"// Total: {total_nodes} nodes, {total_edges} edges")
    cypher_queries.append("// ========================================")
    cypher_queries.append("")
    cypher_queries.append("// The graph is already in your Neo4j database!")
    cypher_queries.append("// Copy and paste these queries into Neo4j Browser (http://localhost:7474)")
    cypher_queries.append("")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// QUICK START: View Full Graph")
    cypher_queries.append("// ========================================")
    cypher_queries.append("MATCH (n:Conversation) RETURN n LIMIT 5000;")
    cypher_queries.append("")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// View with Relationships")
    cypher_queries.append("// ========================================")
    cypher_queries.append("MATCH (a:Conversation)-[r]->(b:Conversation) RETURN a, r, b LIMIT 1000;")
    cypher_queries.append("")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// Find Most Connected Nodes (Hubs)")
    cypher_queries.append("// ========================================")
    cypher_queries.append("MATCH (n:Conversation)-[r]-() RETURN n, count(r) as connections ORDER BY connections DESC LIMIT 20;")
    cypher_queries.append("")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// Search by Text Content")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// MATCH (n:Conversation) WHERE n.text CONTAINS 'your_search_term' RETURN n LIMIT 100;")
    cypher_queries.append("")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// View by Session")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// MATCH (n:Conversation) WHERE n.session_id = 'your_session_id' RETURN n;")
    cypher_queries.append("")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// Find Shortest Path")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// MATCH path = shortestPath((a:Conversation {id: 'node_id_1'})-[*..10]-(b:Conversation {id: 'node_id_2'})) RETURN path;")
    cypher_queries.append("")
    cypher_queries.append("// ========================================")
    cypher_queries.append("// Tips:")
    cypher_queries.append("// - Remove LIMIT to see all nodes (may be slow for large graphs)")
    cypher_queries.append("// - Click nodes in the graph to see their properties")
    cypher_queries.append("// - Use WHERE clauses to filter")
    cypher_queries.append("// - Zoom and pan for navigation")
    cypher_queries.append("// - Use the search box in Neo4j Browser to find specific nodes")
    
    cypher_script = "\n".join(cypher_queries)
    
    return Response(
        content=cypher_script,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=neo4j_browser_queries.cypher"}
    )

@app.get("/export_gephi")
async def export_gephi():
    """
    Export graph in GEXF format for Gephi (best for large graphs).
    Gephi can handle graphs with 100k+ nodes efficiently.
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime
    
    with memory.driver.session() as session:
        # Get all nodes
        nodes_result = session.run("""
            MATCH (c:Conversation)
            RETURN c.id as id, c.text as text, c.timestamp as timestamp, c.session_id as session_id
            ORDER BY c.timestamp DESC
        """)
        
        # Get all edges
        edges_result = session.run("""
            MATCH (a:Conversation)-[r]->(b:Conversation)
            RETURN a.id as source, b.id as target, type(r) as relType, r.weight as weight
        """)
        
        # Create GEXF XML
        gexf = ET.Element("gexf", attrib={"xmlns": "http://www.gexf.net/1.2draft", "version": "1.2"})
        graph = ET.SubElement(gexf, "graph", attrib={"mode": "static", "defaultedgetype": "directed"})
        
        # Attributes
        attributes = ET.SubElement(graph, "attributes", attrib={"class": "node"})
        ET.SubElement(attributes, "attribute", attrib={"id": "0", "title": "text", "type": "string"})
        ET.SubElement(attributes, "attribute", attrib={"id": "1", "title": "session_id", "type": "string"})
        ET.SubElement(attributes, "attribute", attrib={"id": "2", "title": "timestamp", "type": "long"})
        
        # Nodes
        nodes_elem = ET.SubElement(graph, "nodes")
        node_count = 0
        for record in nodes_result:
            node = ET.SubElement(nodes_elem, "node", attrib={"id": str(record['id']), "label": (record.get('text', '')[:50] or str(record['id']))})
            attvalues = ET.SubElement(node, "attvalues")
            ET.SubElement(attvalues, "attvalue", attrib={"for": "0", "value": str(record.get('text', '')[:200] or '')})
            ET.SubElement(attvalues, "attvalue", attrib={"for": "1", "value": str(record.get('session_id', 'default'))})
            ET.SubElement(attvalues, "attvalue", attrib={"for": "2", "value": str(record.get('timestamp', 0))})
            node_count += 1
            if node_count >= 50000:  # Limit for very large graphs
                break
        
        # Edges
        edges_elem = ET.SubElement(graph, "edges")
        edge_count = 0
        for record in edges_result:
            ET.SubElement(edges_elem, "edge", 
                         attrib={
                             "id": f"e{edge_count}",
                             "source": str(record['source']),
                             "target": str(record['target']),
                             "weight": str(record.get('weight', 1.0)),
                             "type": record.get('relType', 'SEMANTIC')
                         })
            edge_count += 1
            if edge_count >= 100000:  # Limit for very large graphs
                break
        
        # Convert to string
        ET.indent(gexf, space="  ")
        gexf_str = ET.tostring(gexf, encoding='unicode', xml_declaration=True)
        
        return Response(
            content=gexf_str,
            media_type="application/xml",
            headers={"Content-Disposition": f"attachment; filename=graph_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gexf"}
        )

@app.post("/import")
async def import_graph(file: UploadFile = File(...)):
    data = await file.read()
    import json
    try:
        payload = json.loads(data)
        # Reconstruct using existing load/save helpers: use a temp approach
        # Expect payload["nodes"], payload["edges"], or payload["graph"] Cytoscape shape
        if "graph" in payload and "nodes" in payload["graph"]:
            # Convert Cytoscape to internal
            g = payload["graph"]
            memory.graph.clear()
            for n in g.get("nodes", []):
                nid = n["data"]["id"]
                attrs = {k: v for k, v in n["data"].items() if k != "id"}
                memory.graph.add_node(nid, **attrs)
            for e in g.get("edges", []):
                d = e["data"]
                memory.graph.add_edge(d["source"], d["target"], type=d.get("type"), weight=d.get("weight", 1.0))
        elif "nodes" in payload and "edges" in payload:
            # Internal format
            import tempfile
            import os
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp.write(data)
            tmp.flush()
            tmp.close()
            memory.load(tmp.name)
            os.unlink(tmp.name)
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/summarize_topic")
async def summarize_topic(topic: str, k: int = 8):
    node_ids = memory.nodes_by_topic(topic, k=k)
    # Create a mock subgraph for compatibility with existing summarization
    subgraph_data = {"nodes": []}
    if node_ids:
        # Get node details from Neo4j
        with memory.driver.session() as session:
            result = session.run("""
                MATCH (c:Conversation)
                WHERE c.id IN $node_ids
                RETURN c.id as id, c.text as text, c.timestamp as timestamp
                ORDER BY c.timestamp
            """, node_ids=node_ids)
            subgraph_data["nodes"] = [{"id": record["id"], "text": record["text"], "timestamp": record["timestamp"]} for record in result]
    
    summary = manager._summarize_subgraph(subgraph_data)
    return {"topic": topic, "summary": summary, "node_ids": node_ids}

@app.post("/evaluate")
async def evaluate_retrieval(payload: Dict[str, Any]):
    """
    Evaluate retrieval metrics for a query.
    Requires: query, k (optional, default 5), ground_truth_node_ids (optional)
    Returns metrics: CRS, Precision@K, Recall@K, GTE, MIU, SCI
    """
    query = payload.get("query", "")
    k = payload.get("k", 5)
    ground_truth_node_ids = payload.get("ground_truth_node_ids", [])
    
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)
    
    # If no ground truth provided, compute metrics that don't require it
    if not ground_truth_node_ids:
        # Compute CRS, GTE, MIU, SCI without ground truth
        import time as _time
        t0 = _time.time()
        subgraph_data, explanation = memory.retrieve_subgraph(query, k=k)
        t1 = _time.time()
        
        retrieved_ids = [n['id'] for n in subgraph_data.get('nodes', [])]
        
        # CRS
        texts = [n['text'] for n in subgraph_data.get('nodes', [])]
        crs = analytics.contextual_relevance_score(query, texts) if texts else 0.0
        
        # GTE
        elapsed_ms = max((t1 - t0) * 1000.0, 1.0)
        with memory.driver.session() as session:
            total_nodes = session.run("MATCH (c:Conversation) RETURN count(c) as count").single()['count']
        gte = total_nodes / elapsed_ms
        
        # MIU
        if retrieved_ids:
            with memory.driver.session() as session:
                connected_count = session.run("""
                    MATCH (c:Conversation)
                    WHERE c.id IN $retrieved_ids
                    AND (c)-[]-()
                    RETURN count(c) as count
                """, retrieved_ids=retrieved_ids).single()['count']
            miu = connected_count / len(retrieved_ids)
        else:
            miu = 0.0
        
        # SCI
        sci = 0.0
        if memory.session_last_node and retrieved_ids:
            with memory.driver.session() as session:
                last_id = list(memory.session_last_node.values())[-1] if memory.session_last_node else None
                if last_id:
                    last_session = session.run("""
                        MATCH (c:Conversation {id: $last_id})
                        RETURN c.session_id as session_id
                    """, last_id=last_id).single()
                    
                    top_session = session.run("""
                        MATCH (c:Conversation {id: $top_id})
                        RETURN c.session_id as session_id
                    """, top_id=retrieved_ids[0]).single()
                    
                    if last_session and top_session:
                        sci = 1.0 if last_session['session_id'] == top_session['session_id'] else 0.0
        
        return JSONResponse({
            "CRS": float(crs),
            "Precision@K": None,  # Requires ground truth
            "Recall@K": None,  # Requires ground truth
            "GTE": float(gte),
            "MIU": float(miu),
            "SCI": float(sci),
            "retrieved_node_ids": retrieved_ids,
            "explanation": explanation,
            "note": "Precision@K and Recall@K require ground_truth_node_ids"
        })
    else:
        # Full evaluation with ground truth
        metrics = memory.evaluate_retrieval(query, k=k, ground_truth_node_ids=ground_truth_node_ids)
        return JSONResponse(metrics)

@app.post("/compress")
async def compress_memory(payload: Dict[str, Any] = None):
    """
    Compress the graph memory using clustering and summarization.
    """
    try:
        result = compressor.compress_graph()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "error"}, status_code=500)

@app.post("/check_integrity")
async def check_integrity(payload: Dict[str, Any]):
    """
    Check integrity of LLM response against memory.
    """
    answer = payload.get("answer", "")
    query = payload.get("query", "")
    explanation = payload.get("explanation", {})
    
    if not answer:
        return JSONResponse({"error": "Answer is required"}, status_code=400)
    
    try:
        result = integrity_checker.check_integrity(answer, query, explanation)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "error"}, status_code=500)

@app.post("/ask_persona")
async def ask_persona(payload: Dict[str, Any]):
    """
    Ask a question as a specific AI persona or get combined answer.
    """
    query = payload.get("query", "")
    persona = payload.get("persona", "combined")
    session_id = payload.get("session_id", "default")
    print(f"[API] /ask_persona called: query='{query[:50]}...', persona='{persona}', session_id='{session_id}'")
    
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)
    
    try:
        result = persona_manager.ask_with_persona(query, persona, session_id)
        print(f"[API] Returning persona result with answer length: {len(result.get('answer', ''))}")
        return JSONResponse(result)
    except Exception as e:
        print(f"[API ERROR] Exception in ask_persona: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "error": str(e), 
            "status": "error",
            "answer": f"Error: {str(e)}",
            "explanation": {},
            "subgraph_node_ids": []
        }, status_code=500)

@app.post("/gapo_query")
async def gapo_query(payload: Dict[str, Any]):
    """
    Query using Graph-Aware Prompt Optimization.
    """
    query = payload.get("query", "")
    k = payload.get("k", 5)
    alpha = payload.get("alpha", 0.5)
    beta = payload.get("beta", 0.3)
    gamma = payload.get("gamma", 0.2)
    
    if not query:
        return JSONResponse({"error": "Query is required"}, status_code=400)
    
    try:
        subgraph_data, explanation = gapo_optimizer.optimize_node_selection(
            query, k=k, alpha=alpha, beta=beta, gamma=gamma
        )
        
        # Summarize subgraph
        memory_context = manager._summarize_subgraph(subgraph_data)
        
        # Get answer using optimized nodes
        if memory_context and memory_context != "No relevant memories found.":
            memory_section = f"""
### Optimized Memory Context (Optional Reference) ###
{memory_context}

Note: Use this context if it's relevant to the query. If the query is about something new or unrelated, answer using your own knowledge base.
"""
        else:
            memory_section = """
Note: No relevant past conversations found. Answer using your own knowledge and intelligence.
"""
        
        prompt = f"""
You are an AI assistant with optimized memory retrieval. You can remember past conversations, but you also have your own extensive knowledge base.

{memory_section}
### Current Query ###
User: {query}

Instructions:
- Answer naturally in your own words, using your intelligence and knowledge base
- If the memory context above is relevant, you may reference it, but don't be limited by it
- If the query is about something new, general knowledge, or unrelated topics, answer normally like any chatbot would
- Be conversational, helpful, and accurate
"""
        
        try:
            aikipedia_config = Config.get_aikipedia_config()
            response = requests.post(
                aikipedia_config["endpoint"],
                json={
                    "user_id": aikipedia_config["user_id"],
                    "message": prompt,
                    "model": aikipedia_config["model"],
                    "history": [],
                    "systemPrompt": ""
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            answer = (
                data.get("answer")
                or data.get("response")
                or data.get("message")
                or str(data)
            )
        except Exception as e:
            answer = f"Error contacting LLM: {e}"
        
        # Store in memory
        memory.add_turn(query, answer, "default")
        
        return JSONResponse({
            "answer": answer,
            "subgraph": subgraph_data,
            "explanation": explanation,
            "optimization_method": "GAPO"
        })
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "error"}, status_code=500)


