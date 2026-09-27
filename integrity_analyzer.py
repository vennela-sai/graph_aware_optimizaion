# integrity_analyzer.py
"""
Memory Integrity Analyzer: Checks if LLM responses are truthful and consistent with memory.
This is Explainable AI × Graph Memory.
"""

from typing import Dict, Any, List
import re
from text_analytics import TextAnalytics
from neo4j_memory import Neo4jGraphMemory


class IntegrityAnalyzer:
    """
    Analyzes LLM responses for contradictions, hallucinations, and unsupported statements.
    """
    
    def __init__(self, memory: Neo4jGraphMemory, analytics: TextAnalytics):
        self.memory = memory
        self.analytics = analytics
    
    def check_integrity(self, answer: str, query: str, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main integrity check function.
        Returns integrity score and list of issues.
        """
        issues = []
        score = 100.0
        
        # 1. Check if answer contradicts retrieved memory
        memory_summary = explanation.get('summary_of_retrieved_memory', '')
        contradiction_score = self._check_contradictions(answer, memory_summary)
        if contradiction_score < 0.7:
            issues.append({
                "type": "Contradiction",
                "description": f"Answer may contradict retrieved memory (similarity: {contradiction_score:.2f})",
                "severity": "high" if contradiction_score < 0.5 else "medium"
            })
            score -= (1 - contradiction_score) * 30
        
        # 2. Check for unsupported claims
        unsupported = self._find_unsupported_claims(answer, memory_summary)
        if unsupported:
            issues.append({
                "type": "Unsupported Claims",
                "description": f"Found {len(unsupported)} potentially unsupported claims",
                "severity": "medium",
                "details": unsupported[:3]  # Show first 3
            })
            score -= len(unsupported) * 5
        
        # 3. Check for hallucinations (specific facts not in memory)
        hallucinations = self._detect_hallucinations(answer, memory_summary)
        if hallucinations:
            issues.append({
                "type": "Potential Hallucination",
                "description": f"Found {len(hallucinations)} statements not supported by memory",
                "severity": "high",
                "details": hallucinations[:3]
            })
            score -= len(hallucinations) * 10
        
        # 4. Check answer relevance to query
        relevance = self.analytics.contextual_relevance_score(query, [answer])
        if relevance < 0.6:
            issues.append({
                "type": "Low Relevance",
                "description": f"Answer may not be relevant to query (relevance: {relevance:.2f})",
                "severity": "medium"
            })
            score -= (0.6 - relevance) * 20
        
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "status": "pass" if score >= 70 else "warning" if score >= 50 else "fail",
            "issues": issues,
            "relevance_score": relevance,
            "contradiction_score": contradiction_score
        }
    
    def _check_contradictions(self, answer: str, memory: str) -> float:
        """Check if answer contradicts memory using semantic similarity."""
        if not memory:
            return 1.0  # No memory to contradict
        
        # High similarity = no contradiction
        similarity = self.analytics.contextual_relevance_score(answer, [memory])
        return similarity
    
    def _find_unsupported_claims(self, answer: str, memory: str) -> List[str]:
        """Find claims in answer that aren't supported by memory."""
        # Extract factual statements (simple heuristic: sentences with numbers, dates, names)
        answer_sentences = re.split(r'[.!?]+', answer)
        memory_lower = memory.lower()
        
        unsupported = []
        for sentence in answer_sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Check if sentence contains specific facts
            has_facts = any(char.isdigit() for char in sentence) or \
                       any(word.istitle() for word in sentence.split() if len(word) > 3)
            
            if has_facts:
                # Check if this fact is mentioned in memory
                sentence_lower = sentence.lower()
                words = set(sentence_lower.split())
                memory_words = set(memory_lower.split())
                
                # If less than 30% word overlap, might be unsupported
                overlap = len(words.intersection(memory_words)) / max(len(words), 1)
                if overlap < 0.3:
                    unsupported.append(sentence[:100])  # Truncate long sentences
        
        return unsupported
    
    def _detect_hallucinations(self, answer: str, memory: str) -> List[str]:
        """Detect potential hallucinations (specific facts not in memory)."""
        # Extract entities and numbers from answer
        answer_analysis = self.analytics.analyze(answer)
        memory_analysis = self.analytics.analyze(memory) if memory else {"entities": [], "topics": []}
        
        answer_entities = set(e['text'].lower() for e in answer_analysis.get('entities', []))
        memory_entities = set(e['text'].lower() for e in memory_analysis.get('entities', []))
        
        # Find entities in answer not in memory
        new_entities = answer_entities - memory_entities
        
        hallucinations = []
        if new_entities:
            # Check if these are significant (proper nouns, numbers)
            for entity in new_entities:
                if entity[0].isupper() or any(char.isdigit() for char in entity):
                    hallucinations.append(f"New entity '{entity}' not found in memory context")
        
        return hallucinations

