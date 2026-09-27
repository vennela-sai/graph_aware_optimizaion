# persona_fusion.py
"""
Multi-Agent Multi-Persona Memory Fusion
Maintains one memory graph but can answer as different AI personas.
"""

from typing import Dict, Any, List, Optional
import requests
from config import Config


class PersonaFusion:
    """
    Manages multiple AI personas and fuses their responses.
    """
    
    PERSONA_MODELS = {
        "gemini": "google/gemini-2.5-flash",
        "qwen": "qwen/qwen-2.5-coder-32b-instruct",
        "codestral": "mistralai/codestral-2501",
        "deepseek": "deepseek/deepseek-chat-v3-0324",
        "claude": "anthropic/claude-3-haiku"
    }
    
    PERSONA_PROMPTS = {
        "gemini": "You are Google's Gemini AI. Answer in a helpful, clear, and concise manner.",
        "qwen": "You are Qwen AI. Focus on advanced coding, math, and reasoning. Be precise and technical.",
        "codestral": "You are Codestral, a high-performance code generation AI. Focus on code quality and best practices.",
        "deepseek": "You are DeepSeek V3. Provide general-purpose reasoning with clear explanations.",
        "claude": "You are Claude 3 Haiku by Anthropic. Be fast, helpful, and accurate."
    }
    
    def __init__(self, memory, llm_manager):
        self.memory = memory
        self.llm_manager = llm_manager
        self.aikipedia_endpoint = Config.AIKIPEDIA_ENDPOINT
        self.aikipedia_user_id = Config.AIKIPEDIA_USER_ID
    
    def ask_with_persona(self, query: str, persona: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Ask a question as a specific persona.
        """
        print(f"[PERSONA] ask_with_persona called: persona={persona}, query='{query[:50]}...'")
        
        try:
            if persona == "combined":
                return self._ask_combined(query, session_id)
            
            if persona not in self.PERSONA_MODELS:
                persona = "gemini"  # Default
            
            model = self.PERSONA_MODELS[persona]
            persona_prompt = self.PERSONA_PROMPTS[persona]
            
            # Retrieve memory (limit to 20 nodes for subgraph visualization)
            print(f"[PERSONA] Step 1: Retrieving memory subgraph...")
            subgraph, explanation = self.memory.retrieve_subgraph(query, k=20)
            print(f"[PERSONA] Retrieved {len(subgraph.get('nodes', []))} nodes")
            
            print(f"[PERSONA] Step 2: Summarizing subgraph...")
            if subgraph.get('nodes') and len(subgraph['nodes']) > 0:
                memory_context = self.llm_manager._summarize_subgraph(subgraph)
            else:
                memory_context = "No relevant memories found."
            
            # Construct persona-specific prompt
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
{persona_prompt}

{memory_section}
### Current Query ###
User: {query}

Instructions:
- Answer naturally in your own words, using your intelligence and knowledge base
- If the memory context above is relevant, you may reference it, but don't be limited by it
- If the query is about something new, general knowledge, or unrelated topics, answer normally like any chatbot would
- Be conversational, helpful, and accurate
- For coding questions, provide proper code examples
"""
            
            print(f"[PERSONA] Step 3: Calling LLM with model {model}...")
            llm_success = False
            answer = None
            try:
                response = requests.post(
                    self.aikipedia_endpoint,
                    json={
                        "user_id": self.aikipedia_user_id,
                        "message": prompt,
                        "model": model,
                        "history": [],
                        "systemPrompt": persona_prompt
                    },
                    timeout=45,  # Reduced timeout
                )
                print(f"[PERSONA] LLM response status: {response.status_code}")
                response.raise_for_status()
                data = response.json()
                print(f"[PERSONA] LLM response keys: {list(data.keys())}")
                answer = (
                    data.get("answer")
                    or data.get("response")
                    or data.get("message")
                    or str(data)
                )
                print(f"[PERSONA] Got answer, length: {len(answer)} chars")
                llm_success = True
            except requests.exceptions.Timeout:
                answer = "Error: LLM request timed out after 45 seconds. Please try again."
                print("[PERSONA ERROR] LLM timeout")
                llm_success = False
            except requests.exceptions.RequestException as e:
                answer = f"Error contacting LLM service: {e}"
                print(f"[PERSONA ERROR] LLM request failed: {e}")
                llm_success = False
            except Exception as e:
                answer = f"Error: {str(e)}"
                print(f"[PERSONA ERROR] Unexpected error: {e}")
                llm_success = False
            
            # Store in memory only if LLM call succeeded
            if llm_success:
                print(f"[PERSONA] Step 4: Adding turn to memory...")
                try:
                    node_id = self.memory.add_turn(query, answer, session_id)
                    print(f"[PERSONA] Added node with ID: {node_id}")
                except Exception as e:
                    print(f"[PERSONA ERROR] Failed to add turn to memory: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[PERSONA] Skipping memory storage due to LLM error")
            
            result = {
                "answer": answer,
                "persona": persona,
                "model": model,
                "error": None if llm_success else "LLM call failed",
                "explanation": {
                    "summary_of_retrieved_memory": memory_context,
                    "retrieval_details": explanation,
                },
                "subgraph_node_ids": [n['id'] for n in subgraph.get('nodes', [])]
            }
            print(f"[PERSONA] Returning result with {len(result.get('subgraph_node_ids', []))} subgraph nodes")
            return result
        except Exception as e:
            print(f"[PERSONA ERROR] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error: {str(e)}",
                "persona": persona,
                "error": str(e),
                "explanation": {},
                "subgraph_node_ids": []
            }
    
    def _ask_combined(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Get answers from multiple personas and combine them optimally.
        """
        # Get answers from top 3 personas
        personas = ["gemini", "qwen", "claude"]
        answers = []
        
        for persona in personas:
            result = self.ask_with_persona(query, persona, f"{session_id}_{persona}")
            # Only include successful responses (no error and answer doesn't start with "Error:")
            answer = result.get("answer", "")
            has_error = result.get("error") is not None
            is_error_message = answer and answer.strip().startswith("Error:")
            if answer and not has_error and not is_error_message:
                answers.append({
                    "persona": persona,
                    "answer": result["answer"]
                })
            else:
                print(f"[PERSONA] Skipping {persona} response due to error")
        
        if not answers:
            return {
                "answer": "Error: Could not get responses from any persona",
                "persona": "combined"
            }
        
        # Combine answers using LLM
        combined_prompt = f"""
        You are synthesizing answers from multiple AI personas. Combine the following answers into one optimal, comprehensive response:
        
        {chr(10).join([f"{ans['persona'].upper()}: {ans['answer']}" for ans in answers])}
        
        Create a unified answer that takes the best aspects from each persona's response.
        """
        
        try:
            response = requests.post(
                self.aikipedia_endpoint,
                json={
                    "user_id": self.aikipedia_user_id,
                    "message": combined_prompt,
                    "model": "google/gemini-2.5-flash",
                    "history": [],
                    "systemPrompt": ""
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            combined_answer = (
                data.get("answer")
                or data.get("response")
                or data.get("message")
                or str(data)
            )
            
            # Store combined answer
            self.memory.add_turn(query, combined_answer, session_id)
            
            return {
                "answer": combined_answer,
                "persona": "combined",
                "sources": [ans["persona"] for ans in answers],
                "individual_answers": answers
            }
        except Exception as e:
            # Fallback: return best single answer
            return {
                "answer": answers[0]["answer"],
                "persona": "combined",
                "fallback": True,
                "sources": [answers[0]["persona"]]
            }

