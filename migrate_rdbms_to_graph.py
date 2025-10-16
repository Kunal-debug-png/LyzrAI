"""
Refactored RDBMS to Graph Migration Script
Uses the new modular architecture for better maintainability and scalability
"""

import json
import logging
from config import get_config
from database import get_rdbms_connector, get_neo4j_connector
from ontology import generate_ontology_from_rdbms, SchemaExtractor
from graph import GraphMigrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main migration workflow"""
    logger.info("=" * 80)
    logger.info("RDBMS TO NEO4J GRAPH MIGRATION")
    logger.info("=" * 80)
    
    # Load configuration
    config = get_config()
    
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Initialize connectors
    logger.info("\n📊 Initializing database connectors...")
    rdbms_connector = get_rdbms_connector(config.rdbms)
    neo4j_connector = get_neo4j_connector(config.neo4j)
    
    try:
        # Test connections
        if not rdbms_connector.test_connection():
            logger.error("Failed to connect to RDBMS")
            return 1
        
        if not neo4j_connector.test_connection():
            logger.error("Failed to connect to Neo4j")
            return 1
        
        logger.info("✅ Database connections established")
        
        # Step 1: Extract schema
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: EXTRACTING RDBMS SCHEMA")
        logger.info("=" * 80)
        
        schema_extractor = SchemaExtractor(rdbms_connector)
        schema = schema_extractor.extract_schema()
        logger.info(f"✓ Extracted schema for {len(schema)} tables")
        
        # Step 2: Generate ontology
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: GENERATING SEMANTIC ONTOLOGY")
        logger.info("=" * 80)
        
        ontology = generate_ontology_from_rdbms(rdbms_connector, config.llm)
        
        # Save ontology
        with open(config.system.ontology_path, "w") as f:
            json.dump(ontology, f, indent=2)
        logger.info(f"✓ Ontology saved to {config.system.ontology_path}")
        logger.info(f"  Domain: {ontology.get('domain', 'Unknown')}")
        logger.info(f"  Concepts: {len(ontology.get('concepts', []))}")
        logger.info(f"  Relationships: {len(ontology.get('relationships', []))}")
        
        # Step 3: Migrate data to Neo4j
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: MIGRATING DATA TO NEO4J")
        logger.info("=" * 80)
        
        migrator = GraphMigrator(rdbms_connector, neo4j_connector, ontology)
        migrator.perform_full_migration(schema, clear_existing=False)
        
        # Step 4: Display statistics
        logger.info("\n" + "=" * 80)
        logger.info("MIGRATION STATISTICS")
        logger.info("=" * 80)
        
        stats = neo4j_connector.get_statistics()
        logger.info(f"\n📊 Graph Statistics:")
        logger.info(f"   Total Nodes: {stats['total_nodes']}")
        logger.info(f"   Total Relationships: {stats['total_relationships']}")
        logger.info(f"   Node Labels: {len(stats['labels'])}")
        logger.info(f"   Relationship Types: {len(stats['relationship_types'])}")
        
        logger.info(f"\n📋 Node Counts by Label:")
        for label, count in stats['node_counts'].items():
            logger.info(f"   • {label}: {count}")
        
        logger.info(f"\n🔗 Relationship Counts by Type:")
        for rel_type, count in stats['relationship_counts'].items():
            logger.info(f"   • {rel_type}: {count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("  1. Run 'python main_graph.py' to start the Q&A system")
        logger.info("  2. Use 'python sync_database.py' for incremental updates")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        rdbms_connector.close()
        neo4j_connector.close()
        logger.info("\n✅ Connections closed")


if __name__ == "__main__":
    import sys
    sys.exit(main())
