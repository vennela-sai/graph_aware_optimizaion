#!/usr/bin/env python3
"""
Script to import all ChatGPT conversations from an export folder.
Processes all JSON files and imports conversations from the last 7 days.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from text_analytics import TextAnalytics
from neo4j_memory import Neo4jGraphMemory
from config import Config
from import_conversations import parse_chatgpt_export, filter_by_date


def process_chatgpt_folder(
    folder_path: str,
    days: int = 30,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
):
    """
    Process all JSON files in a ChatGPT export folder and import conversations.
    
    Args:
        folder_path: Path to the ChatGPT export folder
        days: Number of days to look back (default 7, use 0 for all)
        neo4j_uri: Neo4j URI (uses config if None)
        neo4j_user: Neo4j user (uses config if None)
        neo4j_password: Neo4j password (uses config if None)
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    # Find all JSON files in the folder
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Initialize Neo4j connection
    if neo4j_uri is None:
        neo4j_config = Config.get_neo4j_config()
        neo4j_uri = neo4j_config["uri"]
        neo4j_user = neo4j_config["user"]
        neo4j_password = neo4j_config["password"]
    
    analytics = TextAnalytics()
    memory = Neo4jGraphMemory(
        analytics=analytics,
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        similarity_threshold=Config.SIMILARITY_THRESHOLD
    )
    
    all_conversations = []
    
    # Process each JSON file
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a conversations file (has mapping) or a list of conversations
            if "mapping" in data:
                # Single conversation file
                conversations = parse_chatgpt_export(data)
                all_conversations.extend(conversations)
                print(f"  Found {len(conversations)} conversation turns")
            elif isinstance(data, list):
                # List of conversations
                for conv_data in data:
                    if "mapping" in conv_data:
                        conversations = parse_chatgpt_export(conv_data)
                        all_conversations.extend(conversations)
                print(f"  Found {len([c for c in data if 'mapping' in c])} conversations")
            elif isinstance(data, dict) and any("mapping" in str(v) for v in data.values()):
                # Dictionary of conversations
                for key, conv_data in data.items():
                    if isinstance(conv_data, dict) and "mapping" in conv_data:
                        conversations = parse_chatgpt_export(conv_data)
                        all_conversations.extend(conversations)
                print(f"  Processed dictionary of conversations")
            else:
                print(f"  Skipping {json_file.name} - doesn't appear to be a conversation file")
                
        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON in {json_file.name}: {e}")
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
    
    print(f"\nTotal conversation turns found: {len(all_conversations)}")
    
    # Filter by date
    if days > 0:
        all_conversations = filter_by_date(all_conversations, days)
        print(f"Filtered to {len(all_conversations)} conversations from the past {days} days")
    
    if not all_conversations:
        print("No conversations to import!")
        memory.close()
        return
    
    # Import conversations
    print(f"\nImporting {len(all_conversations)} conversations into Neo4j...")
    imported = 0
    errors = 0
    
    for i, conv in enumerate(all_conversations, 1):
        try:
            query = conv["query"]
            response = conv["response"]
            session_id = conv.get("session_id", f"chatgpt_import_{i}")
            conv_timestamp = conv.get("timestamp")
            
            node_id = memory.add_turn(query, response, session_id, timestamp=conv_timestamp)
            imported += 1
            
            if i % 10 == 0:
                print(f"  Imported {i}/{len(all_conversations)}...")
                
        except Exception as e:
            errors += 1
            print(f"  Error importing conversation {i}: {e}")
    
    memory.close()
    
    print(f"\n[SUCCESS] Import complete!")
    print(f"   Successfully imported: {imported}")
    print(f"   Errors: {errors}")
    print(f"\n[TIP] You can now view your conversations in the UI at http://127.0.0.1:8000/ui")


if __name__ == "__main__":
    import sys
    
    # Default to the export folder if it exists
    export_folder = "6f7db985cb825a81c08037e2d11042923c321fc4f51510fb4d8ada685d760451-2025-11-14-05-48-15-3cc027f9534c4fa6b7adf2b8f1943553"
    
    if len(sys.argv) > 1:
        export_folder = sys.argv[1]
    
    days = 30  # Changed to 30 days (1 month) instead of 7
    if len(sys.argv) > 2:
        days = int(sys.argv[2])
    
    print(f"Processing ChatGPT export folder: {export_folder}")
    print(f"Importing conversations from the past {days} days (last month)\n")
    
    process_chatgpt_folder(export_folder, days=days)

