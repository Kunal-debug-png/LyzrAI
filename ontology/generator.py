"""
Ontology Generation Module
Uses LLMs to generate semantic ontologies from RDBMS schemas
"""

import json
import logging
from typing import Dict, Any, List
from groq import Groq
from config.settings import LLMConfig

logger = logging.getLogger(__name__)


class OntologyGenerator:
    """Uses LLM to generate semantic ontology from RDBMS schema"""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize ontology generator
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = Groq(api_key=config.groq_api_key)
        self.model = config.groq_model
    
    def analyze_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze schema and generate ontology using LLM
        
        Args:
            schema: Database schema dictionary
        
        Returns:
            Generated ontology dictionary
        """
        schema_description = self._format_schema_for_llm(schema)
        
        prompt = f"""You are an ontology expert. Analyze this relational database schema and create a semantic ontology.

SCHEMA:
{schema_description}

Your task:
1. Identify domain concepts and their semantic meaning
2. Detect entity hierarchies (is-a, part-of relationships)
3. Propose meaningful relationship names (use UPPERCASE_WITH_UNDERSCORES format)
4. Identify implicit relationships not defined by foreign keys
5. Suggest which tables represent concepts vs. junction tables
6. Detect many-to-many relationships through junction tables

IMPORTANT RULES:
- Relationship names must be action-oriented verbs in UPPERCASE (e.g., TEACHES, ENROLLED_IN, BELONGS_TO)
- Keep original column names as properties for FK matching - do NOT rename FK columns
- Junction tables should have is_junction: true
- For FK relationships, use column names exactly as they appear in schema

Return a JSON object with this structure:
{{
  "domain": "Brief description of what this database models",
  "concepts": [
    {{
      "table": "original_table_name",
      "concept": "SemanticConceptName",
      "description": "What this represents in the domain",
      "is_junction": false,
      "parent_concepts": [],
      "properties": [
        {{"column": "column_name", "semantic_name": "column_name", "description": "what it represents", "keep_original": true}}
      ]
    }}
  ],
  "relationships": [
    {{
      "from_table": "table1",
      "to_table": "table2",
      "from_column": "fk_column",
      "to_column": "pk_column",
      "relationship_type": "SEMANTIC_VERB",
      "description": "What this relationship means",
      "cardinality": "one-to-many|many-to-many|one-to-one",
      "is_implicit": false
    }}
  ],
  "junction_relationships": [
    {{
      "junction_table": "junction_table_name",
      "left_table": "table1",
      "right_table": "table2",
      "left_fk": "table1_id",
      "right_fk": "table2_id",
      "relationship_type": "CONNECTS_TO",
      "description": "Many to many relationship"
    }}
  ]
}}

Be creative and domain-aware. Use proper semantic naming conventions."""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            result = response.choices[0].message.content
            # Extract JSON from markdown code blocks if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            ontology = json.loads(result)
            logger.info(f"Generated ontology for domain: {ontology.get('domain', 'Unknown')}")
            return ontology
            
        except Exception as e:
            logger.error(f"Error generating ontology: {e}")
            return self._fallback_ontology(schema)
    
    def _format_schema_for_llm(self, schema: Dict[str, Any]) -> str:
        """Format schema in a readable way for LLM"""
        formatted = []
        for table_name, table_info in schema.items():
            cols = ", ".join([f"{col['name']} ({col['type']})" for col in table_info['columns']])
            pk = ", ".join(table_info['primary_key']) if table_info['primary_key'] else "None"
            fks = []
            for fk in table_info['foreign_keys']:
                fks.append(f"{fk['columns']} -> {fk['ref_table']}.{fk['ref_columns']}")
            fk_str = "; ".join(fks) if fks else "None"
            
            formatted.append(f"""
Table: {table_name}
  Columns: {cols}
  Primary Key: {pk}
  Foreign Keys: {fk_str}
""")
        return "\n".join(formatted)
    
    def _fallback_ontology(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback if LLM fails - basic transformation"""
        relationships = []
        junction_rels = []
        
        for table_name, table_info in schema.items():
            is_junction = len(table_info['foreign_keys']) >= 2 and len(table_info['columns']) <= 4
            
            if is_junction and len(table_info['foreign_keys']) >= 2:
                # This is a junction table
                fks = table_info['foreign_keys']
                junction_rels.append({
                    "junction_table": table_name,
                    "left_table": fks[0]['ref_table'],
                    "right_table": fks[1]['ref_table'],
                    "left_fk": fks[0]['columns'][0],
                    "right_fk": fks[1]['columns'][0],
                    "relationship_type": f"CONNECTED_VIA_{table_name.upper()}",
                    "description": f"Many-to-many through {table_name}"
                })
            else:
                # Regular table with FKs
                for fk in table_info['foreign_keys']:
                    relationships.append({
                        "from_table": table_name,
                        "to_table": fk['ref_table'],
                        "from_column": fk['columns'][0],
                        "to_column": fk['ref_columns'][0],
                        "relationship_type": f"REFERENCES_{fk['ref_table'].upper()}",
                        "description": f"References {fk['ref_table']}",
                        "cardinality": "many-to-one",
                        "is_implicit": False
                    })
        
        return {
            "domain": "Generic Domain",
            "concepts": [
                {
                    "table": table_name,
                    "concept": self._to_pascal_case(table_name),
                    "description": f"Represents {table_name}",
                    "is_junction": len(table_info['foreign_keys']) >= 2 and len(table_info['columns']) <= 4,
                    "parent_concepts": [],
                    "properties": [
                        {"column": col['name'], "semantic_name": col['name'], "description": col['name'], "keep_original": True}
                        for col in table_info['columns']
                    ]
                }
                for table_name, table_info in schema.items()
            ],
            "relationships": relationships,
            "junction_relationships": junction_rels
        }
    
    def _to_pascal_case(self, snake_str: str) -> str:
        """Convert snake_case to PascalCase"""
        return ''.join(word.capitalize() for word in snake_str.split('_'))


class SchemaExtractor:
    """Extracts schema information from RDBMS"""
    
    def __init__(self, rdbms_connector):
        """
        Initialize schema extractor
        
        Args:
            rdbms_connector: RDBMSConnector instance
        """
        self.connector = rdbms_connector
    
    def extract_schema(self) -> Dict[str, Any]:
        """
        Extract complete schema including tables, columns, PKs, and FKs
        
        Returns:
            Schema dictionary
        """
        tables = self.connector.get_table_names()
        schema = {}
        
        for table_name in tables:
            schema[table_name] = {
                "columns": [],
                "primary_key": [],
                "foreign_keys": []
            }
            
            # Get column details
            columns = self.connector.get_columns(table_name)
            for col in columns:
                schema[table_name]["columns"].append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default", None)
                })
            
            # Get primary key
            pk_cols = self.connector.get_primary_keys(table_name)
            schema[table_name]["primary_key"] = pk_cols
            
            # Get foreign keys
            foreign_keys = self.connector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                schema[table_name]["foreign_keys"].append({
                    "columns": fk["constrained_columns"],
                    "ref_table": fk["referred_table"],
                    "ref_columns": fk["referred_columns"]
                })
        
        return schema


def generate_ontology_from_rdbms(rdbms_connector, llm_config: LLMConfig) -> Dict[str, Any]:
    """
    High-level function to generate ontology from RDBMS
    
    Args:
        rdbms_connector: RDBMSConnector instance
        llm_config: LLM configuration
    
    Returns:
        Generated ontology
    """
    logger.info("Extracting schema from RDBMS...")
    extractor = SchemaExtractor(rdbms_connector)
    schema = extractor.extract_schema()
    
    logger.info("Generating ontology using AI...")
    generator = OntologyGenerator(llm_config)
    ontology = generator.analyze_schema(schema)
    
    return ontology
