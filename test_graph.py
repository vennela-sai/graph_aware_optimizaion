from neo4j_memory import Neo4jGraphMemory
from text_analytics import TextAnalytics
from config import Config

# Test Neo4j connection
print("Testing Neo4j connection...")
try:
    analytics = TextAnalytics()
    config = Config.get_neo4j_config()
    
    print(f"Connecting to: {config['uri']}")
    print(f"User: {config['user']}")
    
    memory = Neo4jGraphMemory(
        analytics=analytics,
        uri=config['uri'],
        user=config['user'],
        password=config['password']
    )
    
    print("✓ Connection successful!")
    
    result = memory.to_cytoscape()
    print(f"\nGraph stats:")
    print(f"  Nodes: {len(result['nodes'])}")
    print(f"  Edges: {len(result['edges'])}")
    
    if result['nodes']:
        print(f"\nFirst node: {result['nodes'][0]['data'].get('id', 'N/A')}")
    
    memory.close()
    print("\n✓ Test completed successfully!")
    print("\nTo access Neo4j Browser:")
    print("  1. Make sure Neo4j Desktop is running")
    print("  2. Start your database in Neo4j Desktop")
    print("  3. Click 'Open' button next to your database")
    print("  4. Or visit: http://localhost:7474")
    
except Exception as e:
    print(f"\n✗ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Is Neo4j Desktop installed and running?")
    print("  2. Is your database started in Neo4j Desktop?")
    print("  3. Check if port 7687 (Bolt) and 7474 (HTTP) are available")
    print("  4. Verify username/password in config.py")

