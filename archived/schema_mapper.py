"""
⚠️  DEPRECATED - Legacy Schema Mapper

This file is kept for backward compatibility only.

NEW CODE SHOULD USE:
    from ontology import generate_ontology_from_rdbms

The new modular architecture provides:
- Better separation of concerns
- Easier testing and maintenance
- Reusable components

This script will be removed in a future version.
"""

import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from sqlalchemy import inspect, MetaData, ForeignKey
from utils.RDMSConnector import engine
import logging

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Deprecation warning
logging.warning("⚠️  DEPRECATED: schema_mapper.py is deprecated. Use 'from ontology import generate_ontology_from_rdbms' instead.")

class SchemaToOntologyMapper:
    """Converts relational schemas to semantic ontologies using LLMs."""
    
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.3
        )
        self.metadata = MetaData()
        self.metadata.reflect(bind=engine)
    
    def get_schema_structure(self) -> Dict[str, Any]:
        """Extract full schema structure including foreign keys and relationships."""
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        schema = {}
        for table_name in tables:
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)
            
            schema[table_name] = {
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"]
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "column": fk["constrained_columns"][0],
                        "references_table": fk["referred_table"],
                        "references_column": fk["referred_columns"][0]
                    }
                    for fk in foreign_keys
                ]
            }
        
        return schema
    
    def detect_implicit_relationships(self, schema: Dict) -> List[Dict]:
        """Detect implicit relationships and hierarchical dependencies."""
        prompt = f"""
Analyze this relational schema and detect IMPLICIT relationships and hierarchies not explicitly defined by foreign keys:

{json.dumps(schema, indent=2)}

Provide a JSON list of detected relationships with:
1. source_entity
2. target_entity
3. relationship_type (e.g., "HAS", "BELONGS_TO", "DERIVED_FROM", "AGGREGATES", "HIERARCHICAL")
4. confidence_score (0-1)
5. reasoning

Return ONLY valid JSON array, no markdown.
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            relationships = json.loads(response.content)
            return relationships
        except json.JSONDecodeError:
            # Fallback: extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
    
    def generate_ontology(self, schema: Dict, implicit_rels: List[Dict]) -> Dict[str, Any]:
        """Generate semantic ontology from schema."""
        prompt = f"""
Transform this relational schema into a semantic ontology with proper entity classes and relationships:

SCHEMA:
{json.dumps(schema, indent=2)}

IMPLICIT RELATIONSHIPS:
{json.dumps(implicit_rels, indent=2)}

Generate an ontology with:
1. Entity Classes: Map tables to entity classes with semantic labels
2. Attributes: Map columns to entity attributes with data types
3. Relationships: Define relationship types between entities
4. Hierarchies: Identify entity hierarchies (is-a relationships)
5. Semantic Enhancements: Add meaningful labels and descriptions

IMPORTANT: semantic_label must be SINGLE_WORD or SNAKE_CASE (no spaces or special chars)
Examples: USER, PRODUCT, ORDER, USER_PROFILE, PRODUCT_REVIEW

Return ONLY valid JSON with this structure:
{{
    "entity_classes": [
        {{
            "name": "...",
            "from_table": "...",
            "semantic_label": "SINGLE_WORD_OR_SNAKE_CASE",
            "description": "...",
            "attributes": [...],
            "is_abstract": false,
            "parent_class": "..."
        }}
    ],
    "relationships": [
        {{
            "source_class": "...",
            "target_class": "...",
            "relationship_type": "RELATION_TYPE_NO_SPACES",
            "cardinality": "1:1|1:N|M:N",
            "is_implicit": true/false
        }}
    ]
}}
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            ontology = json.loads(response.content)
            
            # Sanitize semantic labels
            for entity in ontology.get("entity_classes", []):
                entity["semantic_label"] = ''.join(
                    c if c.isalnum() else '_' for c in entity.get("semantic_label", "ENTITY")
                ).upper()
                if not entity["semantic_label"]:
                    entity["semantic_label"] = "ENTITY"
            
            # Sanitize relationship types
            for rel in ontology.get("relationships", []):
                rel["relationship_type"] = ''.join(
                    c if c.isalnum() else '_' for c in rel.get("relationship_type", "RELATED_TO")
                ).upper()
                if not rel["relationship_type"]:
                    rel["relationship_type"] = "RELATED_TO"
            
            return ontology
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                ontology = json.loads(json_match.group())
                
                # Sanitize here too
                for entity in ontology.get("entity_classes", []):
                    entity["semantic_label"] = ''.join(
                        c if c.isalnum() else '_' for c in entity.get("semantic_label", "ENTITY")
                    ).upper()
                
                return ontology
            return {"entity_classes": [], "relationships": []}
    
    def map_schema_to_ontology(self) -> Dict[str, Any]:
        """Complete pipeline: schema -> implicit relationships -> ontology."""
        print("📊 Step 1: Extracting schema structure...")
        schema = self.get_schema_structure()
        print(f"   ✅ Found {len(schema)} tables")
        
        print("\n🔍 Step 2: Detecting implicit relationships...")
        implicit_rels = self.detect_implicit_relationships(schema)
        print(f"   ✅ Detected {len(implicit_rels)} implicit relationships")
        
        print("\n🧠 Step 3: Generating semantic ontology...")
        ontology = self.generate_ontology(schema, implicit_rels)
        print(f"   ✅ Generated ontology with {len(ontology.get('entity_classes', []))} classes")
        
        return {
            "schema": schema,
            "implicit_relationships": implicit_rels,
            "ontology": ontology,
            "metadata": {
                "total_tables": len(schema),
                "total_implicit_relations": len(implicit_rels),
                "total_entity_classes": len(ontology.get("entity_classes", []))
            }
        }