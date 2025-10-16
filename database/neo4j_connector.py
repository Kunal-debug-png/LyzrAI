"""
Neo4j Graph Database Connector Module
Handles connections and operations for Neo4j graph database
"""

from typing import Optional, Any, Dict, List
from neo4j import GraphDatabase, Driver, Session
from config.settings import Neo4jConfig
import logging

logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Manages Neo4j database connections and operations"""
    
    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j connector
        
        Args:
            config: Neo4j configuration object
        """
        self.config = config
        self._driver: Optional[Driver] = None
    
    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password)
            )
            logger.info("Neo4j driver created")
        return self._driver
    
    def get_session(self, database: Optional[str] = None) -> Session:
        """
        Get a Neo4j session
        
        Args:
            database: Optional database name, uses default if not provided
        
        Returns:
            Neo4j session
        """
        db = database or self.config.database
        return self.driver.session(database=db)
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                     database: Optional[str] = None) -> List[Dict]:
        """
        Execute a Cypher query and return results
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name
        
        Returns:
            List of result records as dictionaries
        """
        with self.get_session(database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                     database: Optional[str] = None) -> Any:
        """
        Execute a write transaction
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name
        
        Returns:
            Transaction result
        """
        with self.get_session(database) as session:
            return session.write_transaction(
                lambda tx: tx.run(query, parameters or {}).single()
            )
    
    def clear_database(self, database: Optional[str] = None):
        """
        Clear all nodes and relationships from database
        
        Args:
            database: Optional database name
        """
        logger.warning("Clearing Neo4j database...")
        with self.get_session(database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j database cleared")
    
    def get_node_count(self, label: Optional[str] = None, 
                      database: Optional[str] = None) -> int:
        """
        Get count of nodes, optionally filtered by label
        
        Args:
            label: Optional node label to filter
            database: Optional database name
        
        Returns:
            Node count
        """
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"
        
        result = self.execute_query(query, database=database)
        return result[0]["count"] if result else 0
    
    def get_relationship_count(self, rel_type: Optional[str] = None,
                              database: Optional[str] = None) -> int:
        """
        Get count of relationships, optionally filtered by type
        
        Args:
            rel_type: Optional relationship type to filter
            database: Optional database name
        
        Returns:
            Relationship count
        """
        if rel_type:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) as count"
        
        result = self.execute_query(query, database=database)
        return result[0]["count"] if result else 0
    
    def get_labels(self, database: Optional[str] = None) -> List[str]:
        """
        Get all node labels in database
        
        Args:
            database: Optional database name
        
        Returns:
            List of label names
        """
        query = "CALL db.labels()"
        result = self.execute_query(query, database=database)
        return [record["label"] for record in result]
    
    def get_relationship_types(self, database: Optional[str] = None) -> List[str]:
        """
        Get all relationship types in database
        
        Args:
            database: Optional database name
        
        Returns:
            List of relationship type names
        """
        query = "CALL db.relationshipTypes()"
        result = self.execute_query(query, database=database)
        return [record["relationshipType"] for record in result]
    
    def create_constraint(self, label: str, property_name: str,
                         constraint_name: Optional[str] = None,
                         database: Optional[str] = None):
        """
        Create a uniqueness constraint on a node property
        
        Args:
            label: Node label
            property_name: Property name
            constraint_name: Optional constraint name
            database: Optional database name
        """
        if constraint_name is None:
            constraint_name = f"{label}_{property_name}_unique".replace("-", "_")
        
        query = f"""
        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
        FOR (n:{label})
        REQUIRE n.{property_name} IS UNIQUE
        """
        
        try:
            with self.get_session(database) as session:
                session.run(query)
            logger.info(f"Created constraint: {constraint_name}")
        except Exception as e:
            logger.warning(f"Constraint creation warning for {constraint_name}: {e}")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Neo4j connection test successful")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}")
            return False
    
    def get_statistics(self, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database statistics
        
        Args:
            database: Optional database name
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_nodes": self.get_node_count(database=database),
            "total_relationships": self.get_relationship_count(database=database),
            "labels": self.get_labels(database=database),
            "relationship_types": self.get_relationship_types(database=database)
        }
        
        # Get counts per label
        node_counts = {}
        for label in stats["labels"]:
            node_counts[label] = self.get_node_count(label, database=database)
        stats["node_counts"] = node_counts
        
        # Get counts per relationship type
        rel_counts = {}
        for rel_type in stats["relationship_types"]:
            rel_counts[rel_type] = self.get_relationship_count(rel_type, database=database)
        stats["relationship_counts"] = rel_counts
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed")
            self._driver = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Singleton instance for backward compatibility
_default_connector: Optional[Neo4jConnector] = None


def get_neo4j_connector(config: Optional[Neo4jConfig] = None) -> Neo4jConnector:
    """
    Get or create default Neo4j connector
    
    Args:
        config: Optional config, uses default if not provided
    
    Returns:
        Neo4jConnector instance
    """
    global _default_connector
    
    if _default_connector is None:
        if config is None:
            config = Neo4jConfig.from_env()
        _default_connector = Neo4jConnector(config)
    
    return _default_connector
