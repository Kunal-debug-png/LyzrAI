# RDBMS to Knowledge Graph System with Agentic Q&A
Demo : https://drive.google.com/file/d/14GGz42YlQzF_2thsWc8vFmNAI_Zilvmi/view?usp=sharing
A comprehensive system that transforms relational database schemas into semantic knowledge graphs and provides intelligent question-answering capabilities using agentic AI retrieval strategies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Components](#components)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This system bridges the gap between traditional relational databases and modern knowledge graph technologies. It automatically generates semantic ontologies from RDBMS schemas, migrates data to Neo4j, and provides an intelligent Q&A interface powered by large language models and vector embeddings.

### Key Capabilities

- **Automatic Ontology Generation**: Analyzes RDBMS schema and generates semantic ontologies using LLMs
- **Intelligent Data Migration**: Transforms relational data into graph structures with proper relationships
- **Agentic Q&A System**: Uses autonomous agents to select optimal retrieval strategies
- **Multiple Retrieval Strategies**: Vector search, graph traversal, logical filtering, and Cypher queries
- **RESTful API**: Complete FastAPI-based REST API for all operations
- **Interactive CLI**: Command-line interface for exploration and testing

## Features

### Core Features

1. **Schema Analysis and Ontology Generation**
   - Automatic extraction of database schema
   - LLM-powered semantic ontology creation
   - Domain-specific concept and relationship identification

2. **RDBMS to Neo4j Migration**
   - Automated data migration from relational to graph database
   - Relationship inference based on foreign keys
   - Batch processing for large datasets

3. **Agentic Retrieval System**
   - Autonomous strategy selection based on query analysis
   - Vector similarity search using Cohere embeddings
   - Graph traversal for relationship queries
   - Cypher query generation for complex queries
   - Hybrid approach combining multiple strategies

4. **Natural Language Q&A**
   - Streaming responses for real-time interaction
   - Reasoning chain transparency
   - Context-aware answer synthesis
   - Entity exploration capabilities

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                         │
│  (Ontology Generation, Migration, Q&A Endpoints)            │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────┐         ┌─────▼──────┐
│   RDBMS    │         │   Neo4j    │
│ (MySQL    │────────▶│  (Graph    │
│           )│         │  Database) │
└────────────┘         └─────┬──────┘
                             │
                    ┌────────┴────────┐
                    │                 │
            ┌───────▼──────┐   ┌─────▼──────────┐
            │  Ontology    │   │  Vector Store  │
            │  Generator   │   │  (Chroma)      │
            └──────────────┘   └────────────────┘
                                       │
                              ┌────────▼─────────┐
                              │  Agentic Q&A     │
                              │  System          │
                              │  (LLM + RAG)     │
                              └──────────────────┘
```

### Technology Stack

- **Backend Framework**: FastAPI
- **Graph Database**: Neo4j
- **Relational Database**: MySQL/PostgreSQL (via SQLAlchemy)
- **LLM Provider**: Groq (Llama 3.3 70B)
- **Embeddings**: Cohere (embed-english-v3.0)
- **Vector Store**: ChromaDB
- **Language Chain**: LangChain

## Prerequisites

### System Requirements

- Python 3.9 or higher
- Neo4j 5.x (local or cloud instance)
- MySQL or PostgreSQL database
- Minimum 8GB RAM recommended
- Internet connection for API calls

### API Keys Required

1. **Groq API Key**: For LLM operations
   - Sign up at: https://console.groq.com/
   
2. **Cohere API Key**: For embeddings
   - Sign up at: https://dashboard.cohere.com/

3. **Neo4j Credentials**: For graph database
   - Local installation or Neo4j Aura cloud instance

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kunal-debug-png/LyzrAI.git
cd "Lyzr hackathon"
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Neo4j

**Option A: Local Installation**
```bash
# Download and install Neo4j Desktop from neo4j.com
# Create a new database and start it
```

**Option B: Neo4j Aura (Cloud)**
```bash
# Sign up at neo4j.com/cloud/aura
# Create a free instance
# Note down the connection URI and credentials
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (use `.env.example` as template):

```env
# Neo4j Database Configuration
NEO4J_URI=neo4j+s://your_instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# API Keys
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# RDBMS Configuration (MySQL example)
RDBMS_TYPE=mysql
RDBMS_HOST=localhost
RDBMS_PORT=3306
RDBMS_DATABASE=your_database_name
RDBMS_USER=your_database_user
RDBMS_PASSWORD=your_database_password

# Alternative: Direct connection string
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/database_name

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8000
```

### Configuration Files

The system uses centralized configuration management:

- `config/settings.py`: Main configuration classes
- `.env`: Environment-specific variables
- `ontology_output.json`: Generated ontology (created automatically)

## Usage

### Method 1: FastAPI Server (Recommended)

#### Start the Server

```bash
python start_api.py
```

The server will start on `http://localhost:8000`

#### Access API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

#### API Workflow

1. **Generate Ontology**
   ```bash
   POST /api/ontology/generate
   {
     "force_regenerate": false,
     "ontology_path": "ontology_output.json"
   }
   ```

2. **Execute Migration**
   ```bash
   POST /api/migration/execute
   {
     "clear_existing": false,
     "ontology_path": "ontology_output.json"
   }
   ```

3. **Initialize Q&A System**
   ```bash
   POST /api/qa/initialize
   {
     "force_reprocess": false,
     "ontology_path": "ontology_output.json"
   }
   ```

4. **Ask Questions**
   ```bash
   POST /api/qa/ask
   {
     "question": "What are the main entities in the database?",
     "stream": false,
     "initialize_if_needed": true
   }
   ```

### Method 2: Interactive CLI

```bash
python main_graph.py
```

This launches an interactive command-line interface with:
- System information display
- Strategy demonstration
- Example queries
- Interactive Q&A mode

#### CLI Commands

- Type your question to get an answer
- `explain: <question>` - Show strategy without executing
- `explore: <entity>` - Explore an entity's neighborhood
- `stats` - Show graph statistics
- `exit` - Quit the system

### Method 3: Direct Script Execution

#### Generate Ontology Only

```bash
python -c "from ontology import generate_ontology_from_rdbms; from config import get_config; from database import get_rdbms_connector; import json; config = get_config(); rdbms = get_rdbms_connector(config.rdbms); ontology = generate_ontology_from_rdbms(rdbms, config.llm); json.dump(ontology, open('ontology_output.json', 'w'), indent=2)"
```

#### Migrate Data Only

```bash
python migrate_rdbms_to_graph.py
```

## API Documentation

### Ontology Endpoints

#### POST /api/ontology/generate

Generate semantic ontology from RDBMS schema.

**Request Body:**
```json
{
  "force_regenerate": false,
  "ontology_path": "ontology_output.json"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Ontology generated successfully",
  "ontology_path": "ontology_output.json",
  "domain": "E-commerce Platform",
  "concepts_count": 15,
  "relationships_count": 22,
  "ontology": {...}
}
```

#### GET /api/ontology

Retrieve the current ontology.

**Query Parameters:**
- `ontology_path` (optional): Path to ontology file

### Migration Endpoints

#### POST /api/migration/execute

Execute RDBMS to Neo4j graph migration.

**Request Body:**
```json
{
  "clear_existing": false,
  "ontology_path": "ontology_output.json"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Migration completed successfully",
  "statistics": {
    "total_nodes": 1500,
    "total_relationships": 3200,
    "node_counts": {...},
    "relationship_counts": {...}
  }
}
```

### Q&A Endpoints

#### POST /api/qa/initialize

Initialize the Q&A system.

**Request Body:**
```json
{
  "force_reprocess": false,
  "ontology_path": "ontology_output.json"
}
```

#### POST /api/qa/ask

Ask a question to the knowledge graph.

**Request Body:**
```json
{
  "question": "Who are the top customers?",
  "stream": false,
  "initialize_if_needed": true
}
```

**Response (Non-streaming):**
```json
{
  "success": true,
  "question": "Who are the top customers?",
  "answer": "Based on the database...",
  "reasoning_chain": [...],
  "strategy_used": "cypher_query"
}
```

#### POST /api/qa/explain

Explain what strategy would be used for a question.

**Query Parameters:**
- `question`: The question to analyze

#### POST /api/qa/explore

Explore an entity and its neighborhood.

**Request Body:**
```json
{
  "entity_identifier": "customer_123",
  "depth": 2
}
```

#### GET /api/qa/statistics

Get graph statistics.

### Health Endpoint

#### GET /health

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "qa_system_initialized": true,
  "neo4j_connected": true,
  "ontology_exists": true
}
```

## Project Structure

```
Lyzr hackathon/
├── api_main.py                 # FastAPI application
├── start_api.py                # Server startup script
├── main_graph.py               # Interactive CLI application
├── rdbms_graph_qa.py           # Agentic Q&A system
├── migrate_rdbms_to_graph.py   # Migration script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create from .env.example)
├── .env.example                # Environment template
├── ontology_output.json        # Generated ontology (auto-created)
│
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py             # Centralized config classes
│
├── database/                   # Database connectors
│   ├── __init__.py
│   ├── rdbms_connector.py      # RDBMS connection handler
│   └── neo4j_connector.py      # Neo4j connection handler
│
├── ontology/                   # Ontology generation
│   ├── __init__.py
│   └── generator.py            # LLM-based ontology generator
│
├── graph/                      # Graph operations
│   ├── __init__.py
│   └── operations.py           # Migration and graph operations
│
├── embeddings/                 # Vector embeddings
│   └── (generated files)
│
├── graph_artifacts/            # Generated artifacts
│   ├── vector_store.pkl
│   └── graph_metadata.json
│
├── sync/                       # Synchronization utilities
│   └── (sync tracking files)
│
├── utils/                      # Utility functions
│   └── (helper modules)
│
└── archived/                   # Archived/legacy code
    └── (old implementations)
```

## Components

### 1. Configuration System (`config/`)

Centralized configuration management using dataclasses:
- `Neo4jConfig`: Neo4j connection settings
- `RDBMSConfig`: Relational database settings
- `LLMConfig`: LLM API configuration
- `EmbeddingConfig`: Embedding model settings
- `SystemConfig`: System-wide settings

### 2. Database Connectors (`database/`)

#### RDBMS Connector
- Supports MySQL, PostgreSQL via SQLAlchemy
- Connection pooling
- Schema extraction
- Query execution

#### Neo4j Connector
- Neo4j driver wrapper
- Batch operations
- Transaction management
- Statistics retrieval

### 3. Ontology Generator (`ontology/`)

LLM-powered ontology generation:
- Schema analysis
- Concept identification
- Relationship mapping
- Domain understanding
- Semantic enrichment

### 4. Graph Operations (`graph/`)

Migration and graph management:
- Node creation from table rows
- Relationship inference from foreign keys
- Batch processing
- Index creation
- Constraint management

### 5. Agentic Q&A System (`rdbms_graph_qa.py`)

Multi-strategy retrieval system:

#### Retrieval Strategies

1. **Vector Search**: Semantic similarity using embeddings
2. **Graph Traversal**: Multi-hop path finding
3. **Logical Filter**: Property-based filtering
4. **Cypher Query**: Direct database queries
5. **Hybrid**: Combination of multiple strategies

#### Agent Components

- **Query Analyzer**: Determines optimal strategy
- **Retrieval Executor**: Executes selected strategy
- **Answer Synthesizer**: Generates natural language responses
- **Reasoning Chain**: Provides transparency

### 6. FastAPI Application (`api_main.py`)

RESTful API with:
- CORS middleware
- Request validation (Pydantic)
- Error handling
- Streaming responses
- Background tasks
- Lifespan management

## Development

### Running Tests

```bash
# Test Neo4j connection
python test_neo4j.py

# Verify nodes
python verify_nodes.py

# Quick check
python quick_check.py

# Export all nodes
python export_all_nodes.py
```

### Adding New Features

1. **New Retrieval Strategy**
   - Add strategy to `RetrievalStrategy` enum
   - Implement retrieval method in `AgenticRDBMSGraphQA`
   - Update query analyzer prompt

2. **New API Endpoint**
   - Add Pydantic models in `api_main.py`
   - Implement endpoint handler
   - Update API documentation

3. **New Database Support**
   - Add connector in `database/`
   - Update configuration in `config/settings.py`
   - Test with sample database

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions
- Keep functions focused and small
- Use meaningful variable names

### Logging

The system uses Python's logging module:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
```

## Troubleshooting

### Common Issues

#### 1. Connection Errors

**Problem**: Cannot connect to Neo4j
```
Failed to connect to Neo4j
```

**Solution**:
- Verify Neo4j is running
- Check URI format: `neo4j+s://` for Aura, `bolt://` for local
- Verify credentials in `.env`
- Check firewall settings

#### 2. API Key Errors

**Problem**: Invalid API key
```
GROQ_API_KEY not configured
```

**Solution**:
- Verify API keys in `.env` file
- Check for extra spaces or quotes
- Ensure keys are valid and active
- Check API rate limits

#### 3. Import Errors

**Problem**: Module not found
```
ModuleNotFoundError: No module named 'langchain_groq'
```

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

#### 4. Memory Issues

**Problem**: Out of memory during processing
```
MemoryError
```

**Solution**:
- Reduce batch size in configuration
- Process data in smaller chunks
- Increase system RAM
- Use pagination for large queries

#### 5. Ontology Generation Fails

**Problem**: Ontology generation timeout
```
Ontology generation failed
```

**Solution**:
- Check RDBMS connection
- Verify LLM API is accessible
- Reduce schema complexity
- Increase timeout settings

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in `.env`:
```env
LOG_LEVEL=DEBUG
```

### Getting Help

1. Check logs in console output
2. Review API documentation at `/docs`
3. Verify environment variables
4. Test individual components
5. Check Neo4j browser for data

## Performance Optimization

### Database Optimization

1. **Neo4j Indexes**
   - Create indexes on frequently queried properties
   - Use constraints for unique identifiers

2. **Batch Processing**
   - Adjust batch sizes based on available memory
   - Use transactions for bulk operations

3. **Query Optimization**
   - Use EXPLAIN in Neo4j browser
   - Optimize Cypher queries
   - Limit result sets appropriately

### API Optimization

1. **Caching**
   - Cache ontology in memory
   - Reuse vector store
   - Cache frequently accessed data

2. **Async Operations**
   - Use background tasks for long operations
   - Implement proper async/await patterns

3. **Connection Pooling**
   - Configure appropriate pool sizes
   - Reuse database connections

## Security Considerations

1. **API Keys**: Never commit `.env` file to version control
2. **Database Credentials**: Use environment variables
3. **API Access**: Implement authentication if deploying publicly
4. **Input Validation**: All inputs are validated via Pydantic
5. **SQL Injection**: Use parameterized queries
6. **CORS**: Configure appropriately for production

## Deployment

### Docker Deployment (Optional)

```bash
# Build image
docker build -t rdbms-knowledge-graph .

# Run container
docker run -p 8000:8000 --env-file .env rdbms-knowledge-graph
```

### Production Considerations

1. Use production-grade ASGI server (Gunicorn + Uvicorn)
2. Set up reverse proxy (Nginx)
3. Enable HTTPS
4. Configure proper logging
5. Set up monitoring
6. Implement rate limiting
7. Use managed database services

