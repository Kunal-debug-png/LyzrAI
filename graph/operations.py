"""
Graph Operations Module
Handles CRUD operations and data migration for Neo4j graph database
"""

import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, date, time
from sqlalchemy import text
from database.neo4j_connector import Neo4jConnector
from database.rdbms_connector import RDBMSConnector

logger = logging.getLogger(__name__)


class DataConverter:
    """Converts data types between RDBMS and Neo4j"""
    
    @staticmethod
    def convert_value(value: Any) -> Any:
        """
        Convert database values to Neo4j compatible types
        
        Args:
            value: Value to convert
        
        Returns:
            Converted value
        """
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif isinstance(value, int):
            return value
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


class GraphMigrator:
    """Handles migration of data from RDBMS to Neo4j graph"""
    
    def __init__(self, rdbms_connector: RDBMSConnector, 
                 neo4j_connector: Neo4jConnector,
                 ontology: Dict[str, Any]):
        """
        Initialize graph migrator
        
        Args:
            rdbms_connector: RDBMS connector instance
            neo4j_connector: Neo4j connector instance
            ontology: Ontology dictionary
        """
        self.rdbms = rdbms_connector
        self.neo4j = neo4j_connector
        self.ontology = ontology
        self.converter = DataConverter()
    
    def create_ontology_structure(self):
        """Create ontology structure in Neo4j (constraints)"""
        with self.neo4j.get_session() as session:
            # Create constraints for each concept (only for non-junction tables)
            for concept in self.ontology.get('concepts', []):
                if concept.get('is_junction', False):
                    continue
                
                concept_name = concept['concept']
                table_name = concept['table']
                pk_cols = self.rdbms.get_primary_keys(table_name)
                
                for pk_col in pk_cols:
                    self.neo4j.create_constraint(concept_name, pk_col)
            
            logger.info("Ontology structure created in Neo4j")
    
    def _detect_primary_key(self, table_name: str, concept: Dict, properties: Dict) -> Optional[str]:
        """
        Detect primary key column from table name and properties
        
        Args:
            table_name: Name of the table
            concept: Concept definition
            properties: Node properties
            
        Returns:
            Primary key column name or None
        """
        # Check if specified in ontology
        pk_col = concept.get('primary_key')
        if pk_col and pk_col in properties:
            return pk_col
        
        # Singularize table name properly
        singular = table_name
        if table_name.endswith('ies'):
            singular = table_name[:-3] + 'y'  # categories → category
        elif table_name.endswith('ses'):
            singular = table_name[:-2]  # addresses → address
        elif table_name.endswith('s'):
            singular = table_name[:-1]  # products → product
        
        # Try common patterns
        patterns = [
            f"{singular}_id",      # address_id, product_id
            f"{table_name}_id",    # addresses_id (fallback)
            "id"                   # generic id
        ]
        
        for pattern in patterns:
            if pattern in properties:
                return pattern
        
        return None
    
    def _check_similarity_batch(self, session, label: str, properties: Dict, 
                                threshold: float = 0.95, sample_size: int = 100) -> bool:
        """
        Check if similar node exists by comparing properties on a sample batch
        
        Args:
            session: Neo4j session
            label: Node label
            properties: Properties to check
            threshold: Similarity threshold (0-1)
            sample_size: Number of nodes to sample
            
        Returns:
            True if similar node found, False otherwise
        """
        # Get sample of existing nodes
        query = f"""
        MATCH (n:{label})
        RETURN properties(n) as props
        LIMIT {sample_size}
        """
        
        try:
            result = session.run(query)
            existing_nodes = [record['props'] for record in result]
            
            if not existing_nodes:
                return False  # No existing nodes, not a duplicate
            
            # Calculate similarity with each existing node
            for existing_props in existing_nodes:
                similarity = self._calculate_similarity(properties, existing_props)
                if similarity >= threshold:
                    logger.info(f"  Found similar node (similarity: {similarity:.2f}), skipping")
                    return True  # Similar node found
            
            return False  # No similar node found
            
        except Exception as e:
            logger.warning(f"  Similarity check failed: {e}, assuming not duplicate")
            return False
    
    def _calculate_similarity(self, props1: Dict, props2: Dict) -> float:
        """
        Calculate similarity between two property dictionaries
        
        Args:
            props1: First property dict
            props2: Second property dict
            
        Returns:
            Similarity score (0-1)
        """
        # Get common keys (excluding internal keys)
        keys1 = set(k for k in props1.keys() if not k.startswith('_'))
        keys2 = set(k for k in props2.keys() if not k.startswith('_'))
        common_keys = keys1 & keys2
        
        if not common_keys:
            return 0.0
        
        # Count matching values
        matches = 0
        for key in common_keys:
            if props1.get(key) == props2.get(key):
                matches += 1
        
        # Similarity = matching properties / total common properties
        return matches / len(common_keys)
    
    def migrate_nodes(self, schema: Dict[str, Any]):
        """
        Migrate all nodes from RDBMS to Neo4j
        
        Args:
            schema: Database schema dictionary
        """
        logger.info("\n=== Creating Nodes ===")
        concept_map = {c['table']: c for c in self.ontology.get('concepts', [])}
        
        with self.rdbms.engine.connect() as db:
            with self.neo4j.get_session() as neo_session:
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
                    
                    # Create nodes
                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        properties = {}
                        
                        for key, value in row_dict.items():
                            if value is not None:
                                properties[key] = self.converter.convert_value(value)
                        
                        if properties:
                            # Create node with label
                            labels = [concept_name]
                            if concept.get('parent_concepts'):
                                labels.extend(concept['parent_concepts'])
                            
                            label_str = ':'.join(labels)
                            
                            # === 3-TIER DEDUPLICATION STRATEGY ===
                            
                            # TIER 1: Check by ID/Primary Key (fastest)
                            pk_col = self._detect_primary_key(table_name, concept, properties)
                            
                            if pk_col:
                                # Use MERGE on primary key
                                cypher = f"""
                                MERGE (n:{label_str} {{{pk_col}: ${pk_col}}})
                                ON CREATE SET n = $props
                                ON MATCH SET n = $props
                                """
                                neo_session.run(cypher, **{pk_col: properties[pk_col], 'props': properties})
                            else:
                                # TIER 2: No primary key - compute content hash
                                import hashlib
                                import json
                                
                                sorted_props = dict(sorted(properties.items()))
                                content_str = json.dumps(sorted_props, sort_keys=True, default=str)
                                content_hash = hashlib.md5(content_str.encode()).hexdigest()
                                
                                # Check if node with this hash exists
                                check_query = f"""
                                MATCH (n:{label_str} {{_content_hash: $hash}})
                                RETURN count(n) as cnt
                                """
                                result = neo_session.run(check_query, hash=content_hash)
                                exists = result.single()['cnt'] > 0
                                
                                if not exists:
                                    # TIER 3: Hash doesn't exist - check similarity on sample
                                    # (Only for tables without primary keys)
                                    is_duplicate = self._check_similarity_batch(
                                        neo_session, label_str, properties, threshold=0.95
                                    )
                                    
                                    if not is_duplicate:
                                        # Not a duplicate - create new node
                                        properties['_content_hash'] = content_hash
                                        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                                        cypher = f"CREATE (n:{label_str} {{{props_str}}})"
                                        neo_session.run(cypher, **properties)
                    
                    logger.info(f"  ✓ Created {len(rows)} {concept_name} nodes")
    
    def migrate_relationships(self):
        """Migrate all relationships from RDBMS to Neo4j"""
        logger.info("\n=== Creating Relationships ===")
        concept_map = {c['table']: c for c in self.ontology.get('concepts', [])}
        
        with self.neo4j.get_session() as session:
            # Handle direct foreign key relationships
            for rel in self.ontology.get('relationships', []):
                self._create_direct_relationship(session, rel, concept_map)
            
            # Handle junction table relationships (many-to-many)
            with self.rdbms.engine.connect() as db:
                for junction_rel in self.ontology.get('junction_relationships', []):
                    self._create_junction_relationship(session, junction_rel, concept_map, db)
    
    def _create_direct_relationship(self, session, rel: Dict[str, Any], 
                                   concept_map: Dict[str, Any]):
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
            count = record["cnt"] if record else 0
            logger.info(f"  ✓ {from_concept} -{rel_type}-> {to_concept}: {count} relationships")
        except Exception as e:
            logger.error(f"  ✗ Failed to create relationship {rel_type}: {e}")
    
    def _create_junction_relationship(self, session, junction_rel: Dict[str, Any],
                                     concept_map: Dict[str, Any], db):
        """Create many-to-many relationships through junction tables"""
        junction_table = junction_rel.get('junction_table')
        left_table = junction_rel.get('left_table')
        right_table = junction_rel.get('right_table')
        left_fk = junction_rel.get('left_fk')
        right_fk = junction_rel.get('right_fk')
        rel_type = junction_rel.get('relationship_type')
        
        if not all([junction_table, left_table, right_table, left_fk, right_fk, rel_type]):
            return
        
        left_concept = concept_map.get(left_table, {}).get('concept')
        right_concept = concept_map.get(right_table, {}).get('concept')
        
        if not left_concept or not right_concept:
            return
        
        try:
            # Get junction table data
            query = text(f"SELECT {left_fk}, {right_fk} FROM {junction_table}")
            result = db.execute(query)
            rows = result.fetchall()
            
            count = 0
            for row in rows:
                left_id = row[0]
                right_id = row[1]
                
                if left_id is not None and right_id is not None:
                    cypher = f"""
                    MATCH (a:{left_concept}), (b:{right_concept})
                    WHERE a.{left_fk} = $left_id AND b.{right_fk} = $right_id
                    MERGE (a)-[r:{rel_type}]->(b)
                    """
                    session.run(cypher, left_id=left_id, right_id=right_id)
                    count += 1
            
            logger.info(f"  ✓ {left_concept} -{rel_type}-> {right_concept}: {count} relationships")
        except Exception as e:
            logger.error(f"  ✗ Failed to create junction relationship {rel_type}: {e}")
    
    def perform_full_migration(self, schema: Dict[str, Any], clear_existing: bool = True):
        """
        Perform complete migration from RDBMS to Neo4j
        
        Args:
            schema: Database schema dictionary
            clear_existing: Whether to clear existing data
        """
        if clear_existing:
            logger.info("Clearing existing Neo4j database...")
            self.neo4j.clear_database()
        
        self.create_ontology_structure()
        self.migrate_nodes(schema)
        self.migrate_relationships()
        
        logger.info("\n✅ Migration completed successfully!")


class GraphQueryBuilder:
    """Builds Cypher queries for common operations"""
    
    @staticmethod
    def get_node_by_id(label: str, id_property: str, id_value: Any) -> str:
        """Build query to get node by ID"""
        return f"MATCH (n:{label} {{{id_property}: $id_value}}) RETURN n"
    
    @staticmethod
    def create_node(label: str, properties: Dict[str, Any]) -> str:
        """Build query to create node"""
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        return f"CREATE (n:{label} {{{props_str}}}) RETURN n"
    
    @staticmethod
    def update_node(label: str, id_property: str, properties: Dict[str, Any]) -> str:
        """Build query to update node"""
        set_clause = ", ".join([f"n.{k} = ${k}" for k in properties.keys() if k != id_property])
        return f"""
        MATCH (n:{label} {{{id_property}: ${id_property}}})
        SET {set_clause}
        RETURN n
        """
    
    @staticmethod
    def delete_node(label: str, id_property: str) -> str:
        """Build query to delete node"""
        return f"MATCH (n:{label} {{{id_property}: $id_value}}) DETACH DELETE n"
    
    @staticmethod
    def get_neighbors(label: str, id_property: str, depth: int = 1) -> str:
        """Build query to get node neighbors"""
        return f"""
        MATCH (n:{label} {{{id_property}: $id_value}})
        CALL apoc.path.subgraphAll(n, {{maxLevel: {depth}}})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """


class GraphUpdater:
    """Handles incremental updates to the graph"""
    
    def __init__(self, rdbms_connector: RDBMSConnector,
                 neo4j_connector: Neo4jConnector,
                 ontology: Dict[str, Any]):
        """
        Initialize graph updater
        
        Args:
            rdbms_connector: RDBMS connector instance
            neo4j_connector: Neo4j connector instance
            ontology: Ontology dictionary
        """
        self.rdbms = rdbms_connector
        self.neo4j = neo4j_connector
        self.ontology = ontology
        self.converter = DataConverter()
    
    def sync_table(self, table_name: str, concept: Dict[str, Any]):
        """
        Sync a single table incrementally
        
        Args:
            table_name: Name of the table to sync
            concept: Concept definition from ontology
        """
        concept_name = concept['concept']
        is_junction = concept.get('is_junction', False)
        
        if is_junction:
            logger.info(f"  Skipping junction table (handled via relationships)")
            return
        
        with self.rdbms.engine.connect() as db:
            with self.neo4j.get_session() as neo_session:
                # Get primary key columns
                pk_cols = self.rdbms.get_primary_keys(table_name)
                if not pk_cols:
                    logger.warning(f"  No primary key found for {table_name}, skipping")
                    return
                
                # Fetch all current data from RDBMS
                query = text(f"SELECT * FROM {table_name}")
                result = db.execute(query)
                rows = result.fetchall()
                columns = result.keys()
                
                # Get existing node IDs from Neo4j
                pk_col = pk_cols[0]
                existing_ids_query = f"""
                MATCH (n:{concept_name})
                RETURN n.{pk_col} as id
                """
                existing_result = neo_session.run(existing_ids_query)
                existing_ids = {record["id"] for record in existing_result if record["id"] is not None}
                
                # Process rows
                current_ids = set()
                new_count = 0
                updated_count = 0
                
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    properties = {}
                    
                    for key, value in row_dict.items():
                        if value is not None:
                            properties[key] = self.converter.convert_value(value)
                    
                    if not properties or pk_col not in properties:
                        continue
                    
                    node_id = properties[pk_col]
                    current_ids.add(node_id)
                    
                    # Check if node exists
                    if node_id in existing_ids:
                        # Update existing node
                        set_clause = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
                        update_query = f"""
                        MATCH (n:{concept_name} {{{pk_col}: ${pk_col}}})
                        SET {set_clause}
                        RETURN n
                        """
                        neo_session.run(update_query, **properties)
                        updated_count += 1
                    else:
                        # Create new node
                        labels = [concept_name]
                        if concept.get('parent_concepts'):
                            labels.extend(concept['parent_concepts'])
                        
                        label_str = ':'.join(labels)
                        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                        create_query = f"CREATE (n:{label_str} {{{props_str}}})"
                        neo_session.run(create_query, **properties)
                        new_count += 1
                
                # Delete nodes that no longer exist in RDBMS
                deleted_ids = existing_ids - current_ids
                deleted_count = 0
                
                if deleted_ids:
                    for del_id in deleted_ids:
                        delete_query = f"""
                        MATCH (n:{concept_name} {{{pk_col}: $id}})
                        DETACH DELETE n
                        """
                        neo_session.run(delete_query, id=del_id)
                        deleted_count += 1
                
                logger.info(f"  ✓ New: {new_count}, Updated: {updated_count}, Deleted: {deleted_count}")
