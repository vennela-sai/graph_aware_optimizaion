# llm_manager.py

import os
import requests
from typing import Dict, Any

from config import Config

# This class can be extended for Anthropic, Google, etc.
class LLMManager:
    """
    A wrapper-agnostic manager for LLMs that integrates the GraphMemory.
    """
    def __init__(self, memory):
        self.memory = memory
        aikipedia_config = Config.get_aikipedia_config()
        self.aikipedia_endpoint = aikipedia_config["endpoint"]
        self.aikipedia_user_id = aikipedia_config["user_id"]
        self.aikipedia_model = aikipedia_config["model"]

    def _summarize_subgraph(self, subgraph) -> str:
        """
        Uses an LLM to create a concise summary of the retrieved memory nodes.
        """
        if not subgraph.get('nodes'):
            return "No relevant memories found."

        # Sort nodes by timestamp to provide chronological context
        sorted_nodes = sorted(subgraph['nodes'], key=lambda n: n.get('timestamp', 0))
        
        context_str = "\n".join([node['text'] for node in sorted_nodes])
        
        # A smaller, faster model can be used for summarization
        # Limit context length for speed
        if len(context_str) > 2000:
            context_str = context_str[:2000] + "... (truncated)"
        
        prompt = f"""
        Summarize the key points from the following conversation snippets in 2-3 sentences.
        Focus on facts, user preferences, and unresolved questions.

        Conversation Snippets:
        ---
        {context_str}
        ---
        Concise Summary:
        """
        
        try:
            response = requests.post(
                self.aikipedia_endpoint,
                json={
                    "user_id": self.aikipedia_user_id,
                    "message": prompt,
                    "model": self.aikipedia_model,
                    "history": [],
                    "systemPrompt": ""
                },
                timeout=30,  # Reduced timeout for faster failure
            )
            response.raise_for_status()
            data = response.json()
            # Aikipedia returns a JSON with model response under a standard key; try common fields
            return (
                data.get("answer")
                or data.get("response")
                or data.get("message")
                or str(data)
            )
        except Exception as e:
            print(f"Error in summarization: {e}")
            return "Could not summarize memories."


    def ask(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        The main method to process a query, inject memory, get a response, and update the graph.
        """
        print(f"[DEBUG] ask() called with query: {query[:50]}...")
        
        try:
            # 1. Retrieve relevant memory subgraph
            print("[DEBUG] Step 1: Retrieving subgraph...")
            subgraph, explanation = self.memory.retrieve_subgraph(query)
            print(f"[DEBUG] Retrieved {len(subgraph.get('nodes', []))} nodes from memory")
            
            # 2. Summarize subgraph to conserve token space (skip if no nodes for speed)
            print("[DEBUG] Step 2: Summarizing subgraph...")
            if subgraph.get('nodes') and len(subgraph['nodes']) > 0:
                memory_context = self._summarize_subgraph(subgraph)
            else:
                memory_context = "No relevant memories found."
            print(f"[DEBUG] Memory context length: {len(memory_context)} chars")
            
            # 3. Construct the prompt with injected memory
            if memory_context and memory_context != "No relevant memories found.":
                memory_section = f"""
### Memory Context from Past Conversations (Optional Reference) ###
{memory_context}

Note: Use this context if it's relevant to the query. If the query is about something new or unrelated, answer using your own knowledge base. The memory context is supplementary, not required.
"""
            else:
                memory_section = """
Note: No relevant past conversations found. Answer using your own knowledge and intelligence.
"""
            
            prompt = f"""
You are an AI assistant with long-term memory capabilities. You can remember past conversations, but you also have your own extensive knowledge base.

{memory_section}
### Current Query ###
User: {query}

Instructions:
- Answer naturally in your own words, using your intelligence and knowledge base
- If the memory context above is relevant, you may reference it for continuity, but don't be limited by it
- If the query is about something new, general knowledge, or unrelated topics, answer normally like any chatbot would
- Be conversational, helpful, and accurate
- For coding questions, provide proper code examples with clear explanations
- Think independently - the graph memory is for continuity, not a constraint
"""

            # 4. Call the main LLM via Aikipedia
            print(f"[DEBUG] Step 3: Calling LLM at {self.aikipedia_endpoint}...")
            try:
                main_resp = requests.post(
                    self.aikipedia_endpoint,
                    json={
                        "user_id": self.aikipedia_user_id,
                        "message": prompt,
                        "model": self.aikipedia_model,
                        "history": [],
                        "systemPrompt": ""
                    },
                    timeout=45,  # Reduced timeout
                )
                print(f"[DEBUG] LLM response status: {main_resp.status_code}")
                main_resp.raise_for_status()
                main_data = main_resp.json()
                print(f"[DEBUG] LLM response keys: {list(main_data.keys())}")
                final_answer = (
                    main_data.get("answer")
                    or main_data.get("response")
                    or main_data.get("message")
                    or str(main_data)
                )
                print(f"[DEBUG] Got answer, length: {len(final_answer)} chars")
            except requests.exceptions.Timeout:
                final_answer = "Error: LLM request timed out after 45 seconds. Please try again."
                print("[ERROR] LLM timeout")
            except requests.exceptions.RequestException as e:
                final_answer = f"Error contacting LLM service: {e}"
                print(f"[ERROR] LLM request failed: {e}")
            except Exception as e:
                final_answer = f"Error: {str(e)}"
                print(f"[ERROR] Unexpected error: {e}")

            # 5. Add the new turn to memory (closing the loop)
            print("[DEBUG] Step 4: Adding turn to memory...")
            try:
                node_id = self.memory.add_turn(query, final_answer, session_id)
                print(f"[DEBUG] Added node with ID: {node_id}")
            except Exception as e:
                print(f"[ERROR] Failed to add turn to memory: {e}")
                import traceback
                traceback.print_exc()
            
            # 6. Return the full payload with the explanation
            result = {
                "answer": final_answer,
                "explanation": {
                    "summary_of_retrieved_memory": memory_context,
                    "retrieval_details": explanation,
                },
                "subgraph_node_ids": [n['id'] for n in subgraph.get('nodes', [])]
            }
            print(f"[DEBUG] Returning result with {len(result.get('subgraph_node_ids', []))} subgraph nodes")
            return result
        except Exception as e:
            print(f"[ERROR] Fatal error in ask(): {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error: {str(e)}",
                "explanation": {},
                "subgraph_node_ids": []
            }