#!/usr/bin/env python3
"""
Script to import ChatGPT conversations from the past week into Neo4j.
Reads JSON export files from ChatGPT.
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from text_analytics import TextAnalytics
from neo4j_memory import Neo4jGraphMemory
from config import Config


def parse_chatgpt_export(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse ChatGPT export format.
    ChatGPT exports typically have structure:
    {
        "title": "...",
        "create_time": ...,
        "update_time": ...,
        "mapping": {
            "node_id": {
                "message": {
                    "id": "...",
                    "author": {"role": "user" or "assistant"},
                    "content": {"parts": [...]},
                    "create_time": ...
                }
            }
        }
    }
    """
    conversations = []
    
    if "mapping" not in data:
        print("Warning: No 'mapping' found in ChatGPT export")
        return conversations
    
    # Extract messages from mapping
    messages = []
    for node_id, node_data in data.get("mapping", {}).items():
        if "message" in node_data and node_data["message"]:
            msg = node_data["message"]
            role = msg.get("author", {}).get("role", "")
            if role in ["user", "assistant"]:
                content_parts = msg.get("content", {}).get("parts", [])
                content = " ".join(str(part) for part in content_parts if part)
                create_time = msg.get("create_time")
                
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": create_time,
                    "node_id": node_id
                })
    
    # Sort by timestamp
    messages.sort(key=lambda x: x.get("timestamp", 0) or 0)
    
    # Pair user messages with assistant responses
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            user_msg = messages[i]["content"]
            user_time = messages[i]["timestamp"]
            
            # Find next assistant response
            assistant_msg = ""
            assistant_time = user_time
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                assistant_msg = messages[i + 1]["content"]
                assistant_time = messages[i + 1].get("timestamp", user_time)
                i += 2
            else:
                i += 1
            
            if user_msg and assistant_msg:
                conversations.append({
                    "query": user_msg,
                    "response": assistant_msg,
                    "timestamp": user_time or assistant_time,
                    "session_id": data.get("title", "chatgpt_import")
                })
        else:
            i += 1
    
    return conversations


def filter_by_date(conversations: List[Dict[str, Any]], days: int = 7) -> List[Dict[str, Any]]:
    """
    Filter conversations to only include those from the past N days.
    """
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    filtered = []
    
    for conv in conversations:
        conv_time = conv.get("timestamp", 0)
        if isinstance(conv_time, str):
            try:
                dt = datetime.fromisoformat(conv_time.replace('Z', '+00:00'))
                conv_time = dt.timestamp()
            except:
                continue
        
        if conv_time and conv_time >= cutoff_time:
            filtered.append(conv)
    
    return filtered


def import_conversations(
    file_path: str,
    days: int = 7,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
):
    """
    Import ChatGPT conversations from a JSON export file.
    
    Args:
        file_path: Path to the ChatGPT JSON export file
        days: Number of days to look back (default 7, use 0 for all)
        neo4j_uri: Neo4j URI (uses config if None)
        neo4j_user: Neo4j user (uses config if None)
        neo4j_password: Neo4j password (uses config if None)
    """
    # Load JSON file
    print(f"Loading ChatGPT conversations from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse ChatGPT conversations
    conversations = parse_chatgpt_export(data)
    
    print(f"Found {len(conversations)} conversation turns")
    
    # Filter by date
    if days > 0:
        conversations = filter_by_date(conversations, days)
        print(f"Filtered to {len(conversations)} conversations from the past {days} days")
    
    if not conversations:
        print("No conversations to import!")
        return
    
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
    
    # Import conversations
    print(f"\nImporting {len(conversations)} conversations into Neo4j...")
    imported = 0
    errors = 0
    
    for i, conv in enumerate(conversations, 1):
        try:
            query = conv["query"]
            response = conv["response"]
            session_id = conv.get("session_id", f"chatgpt_import_{i}")
            
            # Use the timestamp from the conversation if available
            conv_timestamp = conv.get("timestamp")
            
            node_id = memory.add_turn(query, response, session_id, timestamp=conv_timestamp)
            imported += 1
            
            if i % 10 == 0:
                print(f"  Imported {i}/{len(conversations)}...")
                
        except Exception as e:
            errors += 1
            print(f"  Error importing conversation {i}: {e}")
    
    memory.close()
    
    print(f"\n✅ Import complete!")
    print(f"   Successfully imported: {imported}")
    print(f"   Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Import ChatGPT conversations into Neo4j"
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the ChatGPT JSON export file"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7, use 0 for all)"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Neo4j URI (default: from config)"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=None,
        help="Neo4j user (default: from config)"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Neo4j password (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return
    
    import_conversations(
        file_path=args.file,
        days=args.days,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )


if __name__ == "__main__":
    main()

