# config.py

import os
from typing import Optional

class Config:
    """Configuration settings for the Graph Memory system."""
    
    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # Aikipedia Configuration
    AIKIPEDIA_ENDPOINT: str = "https://backend.aikipedia.workers.dev/chat"
    AIKIPEDIA_USER_ID: str = "0ce7c98c-8c38-4912-b41a-1d6541e8b626"
    AIKIPEDIA_MODEL: str = "google/gemini-2.5-flash"
    
    # Memory Configuration
    SIMILARITY_THRESHOLD: float = 0.82
    DEFAULT_K: int = 5
    
    # Text Analytics Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SPACY_MODEL: str = "en_core_web_sm"
    
    # API Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["*"]
    
    @classmethod
    def get_neo4j_config(cls) -> dict:
        """Get Neo4j connection configuration."""
        return {
            "uri": cls.NEO4J_URI,
            "user": cls.NEO4J_USER,
            "password": cls.NEO4J_PASSWORD
        }
    
    @classmethod
    def get_aikipedia_config(cls) -> dict:
        """Get Aikipedia API configuration."""
        return {
            "endpoint": cls.AIKIPEDIA_ENDPOINT,
            "user_id": cls.AIKIPEDIA_USER_ID,
            "model": cls.AIKIPEDIA_MODEL
        }
