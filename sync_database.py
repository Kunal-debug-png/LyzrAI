"""
Sync RDBMS changes to Neo4j Graph
Run this script after making changes to your RDBMS database
"""

import json
import logging
from config import get_config
from database import get_rdbms_connector, get_neo4j_connector
from embeddings import EmbeddingProcessor
from sync import IncrementalSyncEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main sync workflow"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "INCREMENTAL DATABASE SYNC" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    # Load configuration
    config = get_config()
    
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Load ontology
    logger.info(f"\n📚 Loading ontology from {config.system.ontology_path}")
    try:
        with open(config.system.ontology_path, 'r') as f:
            ontology = json.load(f)
        logger.info(f"✓ Ontology loaded: {ontology.get('domain', 'Unknown')}")
    except FileNotFoundError:
        logger.error(f"❌ Ontology file not found: {config.system.ontology_path}")
        logger.error("   Please run migrate_rdbms_to_graph.py first")
        return 1
    
    # Initialize connectors
    logger.info("\n🔌 Initializing database connectors...")
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
        
        # Initialize sync engine
        sync_engine = IncrementalSyncEngine(
            rdbms_connector=rdbms_connector,
            neo4j_connector=neo4j_connector,
            ontology=ontology
        )
        
        # Perform sync
        sync_result = sync_engine.perform_sync()
        
        # Update embeddings if there were changes
        if sync_result["status"] == "success":
            changed_tables = [c["table"] for c in sync_result["changes"]["changed_tables"]]
            
            if changed_tables:
                logger.info("\n" + "=" * 80)
                logger.info("UPDATING EMBEDDINGS")
                logger.info("=" * 80)
                
                # Initialize embedding processor
                embedding_processor = EmbeddingProcessor(
                    neo4j_connector=neo4j_connector,
                    embedding_config=config.embedding,
                    system_config=config.system,
                    ontology=ontology
                )
                
                # Update embeddings for changed tables
                embedding_processor.update_embeddings_for_tables(changed_tables)
                
                logger.info("\n✅ Embeddings updated successfully!")
        
        # Display final statistics
        logger.info("\n" + "=" * 80)
        logger.info("SYNC SUMMARY")
        logger.info("=" * 80)
        
        changes = sync_result.get("changes", {})
        logger.info(f"\n📊 Statistics:")
        logger.info(f"   Changed tables: {len(changes.get('changed_tables', []))}")
        logger.info(f"   Unchanged tables: {len(changes.get('unchanged_tables', []))}")
        logger.info(f"   New records: +{changes.get('total_new_records', 0)}")
        logger.info(f"   Deleted records: -{changes.get('total_deleted_records', 0)}")
        
        if changes.get('changed_tables'):
            logger.info(f"\n📋 Changed Tables:")
            for change in changes['changed_tables']:
                logger.info(f"   • {change['table']}: {change['count_delta']:+d} records")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ SYNC COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Sync failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        rdbms_connector.close()
        neo4j_connector.close()
        logger.info("\n✅ Connections closed")


if __name__ == "__main__":
    import sys
    sys.exit(main())
