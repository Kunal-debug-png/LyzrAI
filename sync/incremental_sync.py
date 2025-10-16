"""
Incremental Synchronization Module
Handles incremental updates from RDBMS to Neo4j graph
"""

import logging
from typing import Dict, Any, List
from database import RDBMSConnector, Neo4jConnector
from graph import GraphUpdater, GraphMigrator
from embeddings import EmbeddingProcessor
from .change_tracker import ChangeTracker

logger = logging.getLogger(__name__)


class IncrementalSyncEngine:
    """Handles incremental synchronization from RDBMS to Neo4j"""
    
    def __init__(self, rdbms_connector: RDBMSConnector,
                 neo4j_connector: Neo4jConnector,
                 ontology: Dict[str, Any]):
        """
        Initialize incremental sync engine
        
        Args:
            rdbms_connector: RDBMS connector instance
            neo4j_connector: Neo4j connector instance
            ontology: Ontology dictionary
        """
        self.rdbms = rdbms_connector
        self.neo4j = neo4j_connector
        self.ontology = ontology
        self.change_tracker = ChangeTracker()
        self.graph_updater = GraphUpdater(rdbms_connector, neo4j_connector, ontology)
        self.graph_migrator = GraphMigrator(rdbms_connector, neo4j_connector, ontology)
    
    def detect_changes(self) -> Dict[str, Any]:
        """
        Detect all changes across tables
        
        Returns:
            Dictionary with change information
        """
        logger.info("\n=== Detecting Changes in RDBMS ===")
        
        changes = {
            "changed_tables": [],
            "unchanged_tables": [],
            "total_new_records": 0,
            "total_deleted_records": 0
        }
        
        with self.rdbms.engine.connect() as db:
            concept_map = {c['table']: c for c in self.ontology.get('concepts', [])}
            
            for table_name in concept_map.keys():
                change_info = self.change_tracker.detect_table_changes(table_name, db)
                
                if change_info["has_changed"]:
                    changes["changed_tables"].append(change_info)
                    logger.info(f"✓ {table_name}: Changed (Δ {change_info['count_delta']:+d} records)")
                    
                    if change_info['count_delta'] > 0:
                        changes["total_new_records"] += change_info['count_delta']
                    elif change_info['count_delta'] < 0:
                        changes["total_deleted_records"] += abs(change_info['count_delta'])
                else:
                    changes["unchanged_tables"].append(table_name)
                    logger.info(f"  {table_name}: No changes")
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Changed tables: {len(changes['changed_tables'])}")
        logger.info(f"   Unchanged tables: {len(changes['unchanged_tables'])}")
        logger.info(f"   New records: +{changes['total_new_records']}")
        logger.info(f"   Deleted records: -{changes['total_deleted_records']}")
        
        return changes
    
    def sync_changed_tables(self, changed_tables: List[Dict[str, Any]]):
        """
        Sync changed tables incrementally
        
        Args:
            changed_tables: List of changed table information
        """
        logger.info("\n=== Syncing Changed Tables ===")
        concept_map = {c['table']: c for c in self.ontology.get('concepts', [])}
        
        for change_info in changed_tables:
            table_name = change_info["table"]
            if table_name in concept_map:
                self.graph_updater.sync_table(table_name, concept_map[table_name])
                
                # Update tracking
                with self.rdbms.engine.connect() as db:
                    checksum = self.change_tracker.get_table_checksum(table_name, db)
                    row_count = change_info["current_count"]
                    self.change_tracker.update_table_tracking(table_name, checksum, row_count)
    
    def sync_relationships(self):
        """Recreate relationships after node updates"""
        logger.info("\n🔗 Syncing Relationships...")
        
        with self.neo4j.get_session() as session:
            # Delete all existing relationships
            session.run("MATCH ()-[r]->() DELETE r")
            logger.info("  Cleared existing relationships")
        
        # Recreate relationships using migrator
        self.graph_migrator.migrate_relationships()
    
    def perform_sync(self) -> Dict[str, Any]:
        """
        Perform full incremental sync
        
        Returns:
            Dictionary with sync results
        """
        logger.info("\n" + "=" * 80)
        logger.info("INCREMENTAL SYNC STARTED")
        logger.info("=" * 80)
        
        # Detect changes
        changes = self.detect_changes()
        
        if not changes["changed_tables"]:
            logger.info("\n✓ No changes detected. Graph is up to date!")
            return {"status": "no_changes", "changes": changes}
        
        # Sync changed tables
        self.sync_changed_tables(changes["changed_tables"])
        
        # Sync relationships
        self.sync_relationships()
        
        # Mark sync complete
        self.change_tracker.mark_sync_complete()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ INCREMENTAL SYNC COMPLETED")
        logger.info("=" * 80)
        
        return {"status": "success", "changes": changes}


def perform_incremental_update(rdbms_connector: RDBMSConnector,
                               neo4j_connector: Neo4jConnector,
                               ontology: Dict[str, Any],
                               embedding_processor: EmbeddingProcessor = None) -> Dict[str, Any]:
    """
    High-level function to perform incremental update
    
    Args:
        rdbms_connector: RDBMS connector instance
        neo4j_connector: Neo4j connector instance
        ontology: Ontology dictionary
        embedding_processor: Optional embedding processor for updating embeddings
    
    Returns:
        Dictionary with update status and statistics
    """
    # Perform incremental sync
    sync_engine = IncrementalSyncEngine(rdbms_connector, neo4j_connector, ontology)
    sync_result = sync_engine.perform_sync()
    
    if sync_result["status"] == "no_changes":
        return sync_result
    
    # Update embeddings for changed tables if processor provided
    if embedding_processor:
        changed_table_names = [c["table"] for c in sync_result["changes"]["changed_tables"]]
        if changed_table_names:
            logger.info("\n=== Updating Embeddings ===")
            embedding_processor.update_embeddings_for_tables(changed_table_names)
    
    return sync_result
