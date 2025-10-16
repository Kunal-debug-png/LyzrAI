"""
Centralized configuration management for the RDBMS to GraphRAG system.
All configuration settings are managed here.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    
    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Load Neo4j config from environment variables"""
        return cls(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )


@dataclass
class RDBMSConfig:
    """RDBMS database configuration"""
    connection_string: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    
    @classmethod
    def from_env(cls) -> 'RDBMSConfig':
        """Load RDBMS config from environment variables"""
        return cls(
            connection_string=os.getenv(
                "DATABASE_URL",
                "mysql+pymysql://root:1234%40Kunal@localhost:3306/one"
            ),
            echo=os.getenv("DB_ECHO", "False").lower() == "true",
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10"))
        )


@dataclass
class LLMConfig:
    """LLM API configuration"""
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 8000
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load LLM config from environment variables"""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8000"))
        )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    cohere_api_key: str
    model: str = "embed-english-v3.0"
    batch_size: int = 96
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Load embedding config from environment variables"""
        return cls(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model=os.getenv("COHERE_MODEL", "embed-english-v3.0"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "96"))
        )


@dataclass
class SystemConfig:
    """Overall system configuration"""
    ontology_path: str = "ontology_output.json"
    vector_store_path: str = "graph_artifacts/vector_store.pkl"
    graph_metadata_path: str = "graph_artifacts/graph_metadata.json"
    sync_tracking_path: str = "sync_tracking.json"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load system config from environment variables"""
        return cls(
            ontology_path=os.getenv("ONTOLOGY_PATH", "ontology_output.json"),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "graph_artifacts/vector_store.pkl"),
            graph_metadata_path=os.getenv("GRAPH_METADATA_PATH", "graph_artifacts/graph_metadata.json"),
            sync_tracking_path=os.getenv("SYNC_TRACKING_PATH", "sync_tracking.json"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )


class Config:
    """Main configuration class that aggregates all configs"""
    
    def __init__(self):
        self.neo4j = Neo4jConfig.from_env()
        self.rdbms = RDBMSConfig.from_env()
        self.llm = LLMConfig.from_env()
        self.embedding = EmbeddingConfig.from_env()
        self.system = SystemConfig.from_env()
    
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from environment"""
        return cls()
    
    def validate(self) -> bool:
        """Validate that all required configurations are present"""
        errors = []
        
        if not self.neo4j.uri:
            errors.append("NEO4J_URI is required")
        if not self.neo4j.password:
            errors.append("NEO4J_PASSWORD is required")
        if not self.rdbms.connection_string:
            errors.append("DATABASE_URL is required")
        if not self.llm.groq_api_key or self.llm.groq_api_key == "your_groq_api_key":
            errors.append("GROQ_API_KEY is required")
        if not self.embedding.cohere_api_key or self.embedding.cohere_api_key == "your_cohere_api_key":
            errors.append("COHERE_API_KEY is required")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment"""
    global _config
    _config = Config.load()
    return _config
