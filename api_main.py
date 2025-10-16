"""
FastAPI Application for RDBMS to Knowledge Graph System

This API provides endpoints for:
1. Ontology creation from RDBMS schema
2. RDBMS to Neo4j graph migration
3. Natural language Q&A over the knowledge graph
"""

import json
import logging
import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from config import get_config
from database import get_rdbms_connector, get_neo4j_connector
from ontology import generate_ontology_from_rdbms, SchemaExtractor
from graph import GraphMigrator
from archived.process_rdbms_graph import process_rdbms_graph_sync
from rdbms_graph_qa import AgenticRDBMSGraphQA

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for QA system
qa_system_instance = None
graph_metadata_instance = None
vector_store_instance = None


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    logger.info("🚀 Starting FastAPI application...")
    yield
    logger.info("🛑 Shutting down FastAPI application...")
    # Cleanup QA system if initialized
    global qa_system_instance
    if qa_system_instance:
        qa_system_instance.close()


# Initialize FastAPI app
app = FastAPI(
    title="RDBMS Knowledge Graph API",
    description="API for ontology creation, RDBMS to graph migration, and intelligent Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class OntologyRequest(BaseModel):
    """Request model for ontology generation"""
    force_regenerate: bool = Field(
        default=False,
        description="Force regeneration even if ontology exists"
    )
    ontology_path: Optional[str] = Field(
        default="ontology_output.json",
        description="Path to save the ontology file"
    )


class OntologyResponse(BaseModel):
    """Response model for ontology generation"""
    success: bool
    message: str
    ontology_path: str
    domain: Optional[str] = None
    concepts_count: Optional[int] = None
    relationships_count: Optional[int] = None
    ontology: Optional[Dict[str, Any]] = None


class MigrationRequest(BaseModel):
    """Request model for RDBMS to graph migration"""
    clear_existing: bool = Field(
        default=False,
        description="Clear existing graph data before migration"
    )
    ontology_path: Optional[str] = Field(
        default="ontology_output.json",
        description="Path to ontology file to use for migration"
    )


class MigrationResponse(BaseModel):
    """Response model for migration"""
    success: bool
    message: str
    statistics: Optional[Dict[str, Any]] = None


class QuestionRequest(BaseModel):
    """Request model for Q&A"""
    question: str = Field(..., description="Natural language question")
    stream: bool = Field(
        default=False,
        description="Stream the response"
    )
    initialize_if_needed: bool = Field(
        default=True,
        description="Initialize QA system if not already initialized"
    )


class QuestionResponse(BaseModel):
    """Response model for Q&A (non-streaming)"""
    success: bool
    question: str
    answer: str
    reasoning_chain: List[Dict[str, Any]]
    strategy_used: Optional[str] = None


class InitializeQARequest(BaseModel):
    """Request model for QA system initialization"""
    force_reprocess: bool = Field(
        default=False,
        description="Force reprocessing of embeddings"
    )
    ontology_path: Optional[str] = Field(
        default="ontology_output.json",
        description="Path to ontology file"
    )


class InitializeQAResponse(BaseModel):
    """Response model for QA initialization"""
    success: bool
    message: str
    statistics: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    qa_system_initialized: bool
    neo4j_connected: bool
    ontology_exists: bool


class ExploreEntityRequest(BaseModel):
    """Request model for entity exploration"""
    entity_identifier: str = Field(..., description="Entity identifier to explore")
    depth: int = Field(default=2, ge=1, le=5, description="Traversal depth")


class ExploreEntityResponse(BaseModel):
    """Response model for entity exploration"""
    success: bool
    entity: Optional[Dict[str, Any]] = None
    labels: Optional[List[str]] = None
    connected_count: Optional[int] = None
    relationship_types: Optional[List[str]] = None
    error: Optional[str] = None


# ==================== Helper Functions ====================

def get_config_with_validation():
    """Get and validate configuration"""
    config = get_config()
    try:
        config.validate()
        return config
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {str(e)}"
        )


def check_ontology_exists(ontology_path: str = "ontology_output.json") -> bool:
    """Check if ontology file exists"""
    return os.path.exists(ontology_path)


def load_ontology(ontology_path: str = "ontology_output.json") -> Dict:
    """Load ontology from file"""
    if not check_ontology_exists(ontology_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ontology file not found at {ontology_path}. Please generate ontology first."
        )
    
    with open(ontology_path, 'r') as f:
        return json.load(f)


# ==================== API Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RDBMS Knowledge Graph API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    config = get_config()
    
    # Check Neo4j connection
    neo4j_connected = False
    try:
        neo4j_connector = get_neo4j_connector(config.neo4j)
        neo4j_connected = neo4j_connector.test_connection()
        neo4j_connector.close()
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
    
    # Check ontology existence
    ontology_exists = check_ontology_exists()
    
    # Check QA system
    qa_initialized = qa_system_instance is not None
    
    return HealthResponse(
        status="healthy" if neo4j_connected else "degraded",
        qa_system_initialized=qa_initialized,
        neo4j_connected=neo4j_connected,
        ontology_exists=ontology_exists
    )


@app.post("/api/ontology/generate", response_model=OntologyResponse)
async def generate_ontology(request: OntologyRequest):
    """
    Generate semantic ontology from RDBMS schema.
    
    This endpoint:
    1. Connects to the configured RDBMS
    2. Extracts the database schema
    3. Uses LLM to generate a semantic ontology
    4. Saves the ontology to a JSON file
    """
    logger.info("=" * 80)
    logger.info("API: ONTOLOGY GENERATION")
    logger.info("=" * 80)
    
    # Check if ontology exists and force_regenerate is False
    if check_ontology_exists(request.ontology_path) and not request.force_regenerate:
        ontology = load_ontology(request.ontology_path)
        return OntologyResponse(
            success=True,
            message="Ontology already exists. Use force_regenerate=true to regenerate.",
            ontology_path=request.ontology_path,
            domain=ontology.get('domain'),
            concepts_count=len(ontology.get('concepts', [])),
            relationships_count=len(ontology.get('relationships', [])),
            ontology=ontology
        )
    
    # Get configuration
    config = get_config_with_validation()
    
    # Initialize RDBMS connector
    try:
        rdbms_connector = get_rdbms_connector(config.rdbms)
        
        if not rdbms_connector.test_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to connect to RDBMS"
            )
        
        # Extract schema
        logger.info("Extracting RDBMS schema...")
        schema_extractor = SchemaExtractor(rdbms_connector)
        schema = schema_extractor.extract_schema()
        logger.info(f"✓ Extracted schema for {len(schema)} tables")
        
        # Generate ontology
        logger.info("Generating semantic ontology...")
        ontology = generate_ontology_from_rdbms(rdbms_connector, config.llm)
        
        # Save ontology
        with open(request.ontology_path, "w") as f:
            json.dump(ontology, f, indent=2)
        
        logger.info(f"✓ Ontology saved to {request.ontology_path}")
        
        # Cleanup
        rdbms_connector.close()
        
        return OntologyResponse(
            success=True,
            message="Ontology generated successfully",
            ontology_path=request.ontology_path,
            domain=ontology.get('domain'),
            concepts_count=len(ontology.get('concepts', [])),
            relationships_count=len(ontology.get('relationships', [])),
            ontology=ontology
        )
        
    except Exception as e:
        logger.error(f"Ontology generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ontology generation failed: {str(e)}"
        )


@app.get("/api/ontology", response_model=Dict[str, Any])
async def get_ontology(ontology_path: str = "ontology_output.json"):
    """
    Retrieve the current ontology.
    """
    try:
        ontology = load_ontology(ontology_path)
        return {
            "success": True,
            "ontology": ontology,
            "domain": ontology.get('domain'),
            "concepts_count": len(ontology.get('concepts', [])),
            "relationships_count": len(ontology.get('relationships', []))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load ontology: {str(e)}"
        )


@app.post("/api/migration/execute", response_model=MigrationResponse)
async def execute_migration(request: MigrationRequest, background_tasks: BackgroundTasks):
    """
    Execute RDBMS to Neo4j graph migration.
    
    This endpoint:
    1. Loads the ontology
    2. Connects to RDBMS and Neo4j
    3. Migrates data from RDBMS to Neo4j graph
    4. Creates nodes and relationships based on the ontology
    """
    logger.info("=" * 80)
    logger.info("API: RDBMS TO GRAPH MIGRATION")
    logger.info("=" * 80)
    
    # Check if ontology exists
    if not check_ontology_exists(request.ontology_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ontology not found at {request.ontology_path}. Please generate ontology first."
        )
    
    # Load ontology
    ontology = load_ontology(request.ontology_path)
    
    # Get configuration
    config = get_config_with_validation()
    
    try:
        # Initialize connectors
        logger.info("Initializing database connectors...")
        rdbms_connector = get_rdbms_connector(config.rdbms)
        neo4j_connector = get_neo4j_connector(config.neo4j)
        
        # Test connections
        if not rdbms_connector.test_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to connect to RDBMS"
            )
        
        if not neo4j_connector.test_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to connect to Neo4j"
            )
        
        logger.info("✅ Database connections established")
        
        # Extract schema
        logger.info("Extracting RDBMS schema...")
        schema_extractor = SchemaExtractor(rdbms_connector)
        schema = schema_extractor.extract_schema()
        logger.info(f"✓ Extracted schema for {len(schema)} tables")
        
        # Perform migration
        logger.info("Migrating data to Neo4j...")
        migrator = GraphMigrator(rdbms_connector, neo4j_connector, ontology)
        migrator.perform_full_migration(schema, clear_existing=request.clear_existing)
        
        # Get statistics
        stats = neo4j_connector.get_statistics()
        
        # Cleanup
        rdbms_connector.close()
        neo4j_connector.close()
        
        logger.info("✅ Migration completed successfully")
        
        return MigrationResponse(
            success=True,
            message="Migration completed successfully",
            statistics={
                "total_nodes": stats['total_nodes'],
                "total_relationships": stats['total_relationships'],
                "node_counts": stats['node_counts'],
                "relationship_counts": stats['relationship_counts']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {str(e)}"
        )


@app.post("/api/qa/initialize", response_model=InitializeQAResponse)
async def initialize_qa_system(request: InitializeQARequest):
    """
    Initialize the Q&A system.
    
    This endpoint:
    1. Loads the ontology
    2. Processes the graph and creates embeddings
    3. Initializes the agentic Q&A system
    """
    global qa_system_instance, graph_metadata_instance, vector_store_instance
    
    logger.info("=" * 80)
    logger.info("API: INITIALIZING Q&A SYSTEM")
    logger.info("=" * 80)
    
    # Check if ontology exists
    if not check_ontology_exists(request.ontology_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ontology not found at {request.ontology_path}. Please generate ontology first."
        )
    
    # Get configuration
    config = get_config_with_validation()
    
    try:
        # Get API keys from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")
        
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY not configured in environment"
            )
        
        if not cohere_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="COHERE_API_KEY not configured in environment"
            )
        
        # Process graph and create embeddings
        logger.info("Processing graph and creating embeddings...")
        vector_store, graph_metadata, content_hash = process_rdbms_graph_sync(
            neo4j_uri=config.neo4j.uri,
            neo4j_user=config.neo4j.username,
            neo4j_password=config.neo4j.password,
            cohere_api_key=cohere_api_key,
            ontology_path=request.ontology_path,
            force_reprocess=request.force_reprocess
        )
        
        # Initialize QA system
        logger.info("Initializing agentic Q&A system...")
        qa_system = AgenticRDBMSGraphQA(
            neo4j_uri=config.neo4j.uri,
            neo4j_user=config.neo4j.username,
            neo4j_password=config.neo4j.password,
            groq_api_key=groq_api_key,
            vector_store=vector_store,
            graph_metadata=graph_metadata
        )
        
        # Store in global state
        qa_system_instance = qa_system
        graph_metadata_instance = graph_metadata
        vector_store_instance = vector_store
        
        # Get statistics
        stats = qa_system.get_statistics()
        
        logger.info("✅ Q&A system initialized successfully")
        
        return InitializeQAResponse(
            success=True,
            message="Q&A system initialized successfully",
            statistics={
                "total_nodes": stats.get('total_nodes', 0),
                "total_relationships": stats.get('total_relationships', 0),
                "node_types": len(stats.get('node_counts', {})),
                "relationship_types": len(stats.get('relationship_counts', {})),
                "content_hash": content_hash
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A initialization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A initialization failed: {str(e)}"
        )


@app.post("/api/qa/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the knowledge graph.
    
    This endpoint:
    1. Analyzes the question using LLM
    2. Determines the best retrieval strategy
    3. Executes the retrieval
    4. Synthesizes an answer
    
    Supports both streaming and non-streaming responses.
    """
    global qa_system_instance
    
    # Check if QA system is initialized
    if qa_system_instance is None:
        if request.initialize_if_needed:
            # Auto-initialize
            logger.info("Q&A system not initialized. Auto-initializing...")
            try:
                init_request = InitializeQARequest(force_reprocess=False)
                await initialize_qa_system(init_request)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Q&A system not initialized and auto-initialization failed: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Q&A system not initialized. Please call /api/qa/initialize first."
            )
    
    try:
        if request.stream:
            # Streaming response
            async def generate():
                try:
                    for chunk in qa_system_instance.ask(request.question, stream=True):
                        yield chunk
                except Exception as e:
                    yield f"\n\n❌ Error: {str(e)}"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain"
            )
        else:
            # Non-streaming response
            answer = qa_system_instance.ask(request.question, stream=False)
            reasoning_chain = qa_system_instance.get_reasoning_chain()
            
            # Extract strategy from reasoning chain
            strategy_used = None
            for step in reasoning_chain:
                if "strategy" in step.get("step", "").lower():
                    strategy_used = step.get("details", {}).get("primary_strategy")
                    break
            
            return QuestionResponse(
                success=True,
                question=request.question,
                answer=answer,
                reasoning_chain=reasoning_chain,
                strategy_used=strategy_used
            )
            
    except Exception as e:
        logger.error(f"Question answering failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {str(e)}"
        )


@app.post("/api/qa/explain")
async def explain_strategy(question: str):
    """
    Explain what strategy would be used for a question without executing it.
    """
    global qa_system_instance
    
    if qa_system_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Q&A system not initialized. Please call /api/qa/initialize first."
        )
    
    try:
        explanation = qa_system_instance.explain_strategy(question)
        return {
            "success": True,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Strategy explanation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy explanation failed: {str(e)}"
        )


@app.post("/api/qa/explore", response_model=ExploreEntityResponse)
async def explore_entity(request: ExploreEntityRequest):
    """
    Explore an entity and its neighborhood in the graph.
    """
    global qa_system_instance
    
    if qa_system_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Q&A system not initialized. Please call /api/qa/initialize first."
        )
    
    try:
        result = qa_system_instance.explore_entity(
            request.entity_identifier,
            depth=request.depth
        )
        
        if "error" in result:
            return ExploreEntityResponse(
                success=False,
                error=result["error"]
            )
        
        return ExploreEntityResponse(
            success=True,
            entity=result.get("entity"),
            labels=result.get("labels"),
            connected_count=result.get("connected_count"),
            relationship_types=result.get("relationship_types")
        )
        
    except Exception as e:
        logger.error(f"Entity exploration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity exploration failed: {str(e)}"
        )


@app.get("/api/qa/statistics")
async def get_statistics():
    """
    Get graph statistics.
    """
    global qa_system_instance
    
    if qa_system_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Q&A system not initialized. Please call /api/qa/initialize first."
        )
    
    try:
        stats = qa_system_instance.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# ==================== Run Application ====================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "api_main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
