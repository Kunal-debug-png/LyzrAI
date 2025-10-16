"""
⚠️  DEPRECATED - Legacy Migration Script

This file is kept for backward compatibility only.

NEW CODE SHOULD USE:
    python migrate_rdbms_to_graph.py

The new modular architecture provides:
- Better maintainability
- Proper separation of concerns
- Easier testing
- Scalability

This script will be removed in a future version.
"""

from sqlalchemy import inspect, text
from neo4j import GraphDatabase
from utils.RDMSConnector import get_db, engine
import logging
import json
from groq import Groq
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deprecation warning
logger.warning("⚠️  DEPRECATED: test_sql.py is deprecated. Use 'python migrate_rdbms_to_graph.py' instead.")

class OntologyGenerator:
    """Uses LLM to generate semantic ontology from RDBMS schema"""
    
    def __init__(self, groq_api_key=None):
        self.client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
    
    def analyze_schema(self, schema):
        """Analyze schema and generate ontology using LLM"""
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
                temperature=0.3,
                max_tokens=8000
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
    
    def _format_schema_for_llm(self, schema):
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
    
    def _fallback_ontology(self, schema):
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
    
    def _to_pascal_case(self, snake_str):
        """Convert snake_case to PascalCase"""
        return ''.join(word.capitalize() for word in snake_str.split('_'))


class RDBMSToOntologyConverter:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, groq_api_key=None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.inspector = inspect(engine)
        self.ontology_gen = OntologyGenerator(groq_api_key)
        self.ontology = None
        
    def close(self):
        self.driver.close()
    
    def extract_schema(self):
        """Extract complete schema including tables, columns, PKs, and FKs"""
        tables = self.inspector.get_table_names()
        schema = {}
        
        for table_name in tables:
            schema[table_name] = {
                "columns": [],
                "primary_key": [],
                "foreign_keys": []
            }
            
            # Get column details
            columns = self.inspector.get_columns(table_name)
            for col in columns:
                schema[table_name]["columns"].append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default", None)
                })
            
            # Get primary key
            pk_constraint = self.inspector.get_pk_constraint(table_name)
            if pk_constraint and pk_constraint.get("constrained_columns"):
                schema[table_name]["primary_key"] = pk_constraint["constrained_columns"]
            
            # Get foreign keys
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                schema[table_name]["foreign_keys"].append({
                    "columns": fk["constrained_columns"],
                    "ref_table": fk["referred_table"],
                    "ref_columns": fk["referred_columns"]
                })
        
        return schema
    
    def generate_ontology(self, schema):
        """Generate ontology from schema using LLM"""
        logger.info("Generating ontology using AI...")
        self.ontology = self.ontology_gen.analyze_schema(schema)
        
        # Save ontology to file
        with open("ontology_output.json", "w") as f:
            json.dump(self.ontology, f, indent=2)
        logger.info("Ontology saved to ontology_output.json")
        
        return self.ontology
    
    def create_ontology_in_neo4j(self):
        """Create ontology structure in Neo4j"""
        with self.driver.session() as session:
            # Clear existing data
            logger.info("Clearing existing Neo4j database...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create constraints for each concept (only for non-junction tables)
            for concept in self.ontology['concepts']:
                if concept.get('is_junction', False):
                    continue
                    
                concept_name = concept['concept']
                table_name = concept['table']
                pk_cols = self._get_pk_columns(table_name)
                
                for pk_col in pk_cols:
                    try:
                        constraint_name = f"{concept_name}_{pk_col}_unique".replace("-", "_")
                        session.run(f"""
                            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                            FOR (n:{concept_name})
                            REQUIRE n.{pk_col} IS UNIQUE
                        """)
                        logger.info(f"  Created constraint for {concept_name}.{pk_col}")
                    except Exception as e:
                        logger.warning(f"Constraint creation warning: {e}")
            
            logger.info("Ontology structure created in Neo4j")
    
    def _get_pk_columns(self, table_name):
        """Helper to get primary key columns for a table"""
        pk_constraint = self.inspector.get_pk_constraint(table_name)
        return pk_constraint.get("constrained_columns", []) if pk_constraint else []
    
    def _convert_value(self, value):
        """Convert database values to Neo4j compatible types - keeping types consistent"""
        from decimal import Decimal
        from datetime import datetime, date, time
        
        if value is None:
            return None
        elif isinstance(value, bool):
            return value  # Keep as boolean
        elif isinstance(value, int):
            return value  # Keep as integer
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (datetime, date, time)):
            return value.isoformat()
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        elif isinstance(value, (list, dict)):
            return str(value)
        else:
            return value
    
    def migrate_data_with_ontology(self, schema):
        """Migrate data using the generated ontology"""
        with engine.connect() as db:
            with self.driver.session() as neo_session:
                # Build concept lookup
                concept_map = {c['table']: c for c in self.ontology['concepts']}
                
                # First pass: Create all nodes (skip junction tables)
                logger.info("\n=== Creating Nodes ===")
                for table_name, table_info in schema.items():
                    if table_name not in concept_map:
                        continue
                        
                    concept = concept_map[table_name]
                    concept_name = concept['concept']
                    
                    # Skip junction tables
                    if concept.get('is_junction', False):
                        logger.info(f"Skipping junction table: {table_name}")
                        continue
                    
                    logger.info(f"Migrating '{table_name}' as '{concept_name}'...")
                    
                    # Fetch data from RDBMS
                    query = text(f"SELECT * FROM {table_name}")
                    result = db.execute(query)
                    rows = result.fetchall()
                    columns = result.keys()
                    
                    # Create nodes - KEEP ORIGINAL COLUMN NAMES for FK matching
                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        properties = {}
                        
                        for key, value in row_dict.items():
                            if value is not None:
                                # Always use original column name for properties
                                properties[key] = self._convert_value(value)
                        
                        if properties:
                            # Create node with label
                            labels = [concept_name]
                            if concept.get('parent_concepts'):
                                labels.extend(concept['parent_concepts'])
                            
                            label_str = ':'.join(labels)
                            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                            cypher = f"CREATE (n:{label_str} {{{props_str}}})"
                            neo_session.run(cypher, **properties)
                    
                    logger.info(f"  ✓ Created {len(rows)} {concept_name} nodes")
                
                # Second pass: Create relationships from ontology
                logger.info("\n=== Creating Relationships ===")
                
                # Handle direct foreign key relationships
                for rel in self.ontology.get('relationships', []):
                    self._create_direct_relationship(neo_session, rel, concept_map)
                
                # Handle junction table relationships (many-to-many)
                for junction_rel in self.ontology.get('junction_relationships', []):
                    self._create_junction_relationship(neo_session, junction_rel, concept_map, db)
    
    def _create_direct_relationship(self, session, rel, concept_map):
        """Create direct FK relationships"""
        from_table = rel.get('from_table')
        to_table = rel.get('to_table')
        from_col = rel.get('from_column')
        to_col = rel.get('to_column')
        rel_type = rel.get('relationship_type')
        
        if not all([from_table, to_table, from_col, to_col, rel_type]):
            logger.warning(f"Incomplete relationship definition: {rel}")
            return
        
        from_concept = concept_map.get(from_table, {}).get('concept')
        to_concept = concept_map.get(to_table, {}).get('concept')
        
        if not from_concept or not to_concept:
            return
        
        # Skip if from_table is a junction table
        if concept_map.get(from_table, {}).get('is_junction', False):
            return
        
        try:
            cypher = f"""
            MATCH (a:{from_concept}), (b:{to_concept})
            WHERE a.{from_col} = b.{to_col} 
              AND a.{from_col} IS NOT NULL
              AND b.{to_col} IS NOT NULL
            MERGE (a)-[r:{rel_type}]->(b)
            RETURN count(r) as cnt
            """
            
            result = session.run(cypher)
            record = result.single()
            count = record['cnt'] if record else 0
            
            if count > 0:
                logger.info(f"  ✓ {from_concept} -[{rel_type}]-> {to_concept}: {count} relationships")
            else:
                logger.warning(f"  ⚠ No matches for {from_concept}.{from_col} -> {to_concept}.{to_col}")
        except Exception as e:
            logger.error(f"  ✗ Error creating relationship {rel_type}: {e}")
    
    def _create_junction_relationship(self, session, junction_rel, concept_map, db):
        """Create many-to-many relationships through junction tables"""
        junction_table = junction_rel.get('junction_table')
        left_table = junction_rel.get('left_table')
        right_table = junction_rel.get('right_table')
        left_fk = junction_rel.get('left_fk')
        right_fk = junction_rel.get('right_fk')
        rel_type = junction_rel.get('relationship_type')
        
        if not all([junction_table, left_table, right_table, left_fk, right_fk, rel_type]):
            logger.warning(f"Incomplete junction relationship: {junction_rel}")
            return
        
        left_concept = concept_map.get(left_table, {}).get('concept')
        right_concept = concept_map.get(right_table, {}).get('concept')
        
        if not left_concept or not right_concept:
            return
        
        # Get PK columns for the referenced tables
        left_pk = self._get_pk_columns(left_table)[0] if self._get_pk_columns(left_table) else 'id'
        right_pk = self._get_pk_columns(right_table)[0] if self._get_pk_columns(right_table) else 'id'
        
        try:
            # Fetch junction data
            query = text(f"SELECT {left_fk}, {right_fk} FROM {junction_table}")
            result = db.execute(query)
            
            count = 0
            for row in result:
                left_val, right_val = row
                if left_val is None or right_val is None:
                    continue
                
                cypher = f"""
                MATCH (a:{left_concept}), (b:{right_concept})
                WHERE a.{left_pk} = $left_val AND b.{right_pk} = $right_val
                MERGE (a)-[r:{rel_type}]->(b)
                """
                session.run(cypher, left_val=left_val, right_val=right_val)
                count += 1
            
            if count > 0:
                logger.info(f"  ✓ {left_concept} -[{rel_type}]-> {right_concept}: {count} relationships (via {junction_table})")
        except Exception as e:
            logger.error(f"  ✗ Error creating junction relationship {rel_type}: {e}")
    
    def verify_migration(self):
        """Verify the migration by printing statistics"""
        with self.driver.session() as session:
            # Count nodes by concept
            result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY label")
            logger.info("\n=== Node Counts ===")
            for record in result:
                logger.info(f"  {record['label']}: {record['count']}")
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY type")
            logger.info("\n=== Relationship Counts ===")
            total_rels = 0
            for record in result:
                count = record['count']
                total_rels += count
                logger.info(f"  {record['type']}: {count}")
            
            if total_rels == 0:
                logger.warning("\n⚠ WARNING: No relationships were created!")
            
            # Show sample relationships
            result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN labels(n)[0] as from, type(r) as rel, labels(m)[0] as to, 
                       id(n) as from_id, id(m) as to_id
                LIMIT 10
            """)
            logger.info("\n=== Sample Relationships ===")
            for record in result:
                logger.info(f"  ({record['from']}) -[{record['rel']}]-> ({record['to']})")


def main():
    # Configuration - Load from environment
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    converter = RDBMSToOntologyConverter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GROQ_API_KEY)
    
    try:
        # Step 1: Extract schema
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting RDBMS Schema")
        logger.info("=" * 60)
        schema = converter.extract_schema()
        logger.info(f"Found {len(schema)} tables\n")
        
        # Step 2: Generate ontology using AI
        logger.info("=" * 60)
        logger.info("STEP 2: Generating Semantic Ontology with AI")
        logger.info("=" * 60)
        ontology = converter.generate_ontology(schema)
        logger.info(f"\nDomain: {ontology.get('domain', 'Unknown')}")
        logger.info(f"Concepts: {len(ontology.get('concepts', []))}")
        logger.info(f"Direct Relationships: {len(ontology.get('relationships', []))}")
        logger.info(f"Junction Relationships: {len(ontology.get('junction_relationships', []))}\n")
        
        # Step 3: Create ontology structure in Neo4j
        logger.info("=" * 60)
        logger.info("STEP 3: Creating Ontology Structure in Neo4j")
        logger.info("=" * 60)
        converter.create_ontology_in_neo4j()
        
        # Step 4: Migrate data with semantic mapping
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Migrating Data with Semantic Mapping")
        logger.info("=" * 60)
        converter.migrate_data_with_ontology(schema)
        
        # Step 5: Verify
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Verification")
        logger.info("=" * 60)
        converter.verify_migration()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ONTOLOGY CREATION AND MIGRATION COMPLETED!")
        logger.info("=" * 60)
        logger.info("Check 'ontology_output.json' for the generated ontology\n")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
    finally:
        converter.close()


if __name__ == "__main__":
    main()