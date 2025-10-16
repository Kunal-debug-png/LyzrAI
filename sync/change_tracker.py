"""
Change Tracking Module
Tracks changes in RDBMS tables for incremental synchronization
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any
from datetime import datetime
from sqlalchemy import text

logger = logging.getLogger(__name__)


class ChangeTracker:
    """Tracks changes in RDBMS tables for incremental sync"""
    
    def __init__(self, tracking_file: str = "sync_tracking.json"):
        """
        Initialize change tracker
        
        Args:
            tracking_file: Path to tracking data file
        """
        self.tracking_file = tracking_file
        self.tracking_data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load previous sync tracking data"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load tracking data: {e}")
        
        return {
            "last_sync": None,
            "table_checksums": {},
            "table_row_counts": {},
            "table_last_modified": {}
        }
    
    def _save_tracking_data(self):
        """Save tracking data to file"""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
            logger.info(f"✓ Saved sync tracking data")
        except Exception as e:
            logger.error(f"Failed to save tracking data: {e}")
    
    def get_table_checksum(self, table_name: str, db_connection) -> str:
        """
        Calculate checksum for table data
        
        Args:
            table_name: Name of the table
            db_connection: Database connection
        
        Returns:
            MD5 checksum of table data
        """
        try:
            query = text(f"SELECT * FROM {table_name} ORDER BY 1")
            result = db_connection.execute(query)
            rows = result.fetchall()
            
            # Create hash of all row data
            data_str = str(rows)
            checksum = hashlib.md5(data_str.encode()).hexdigest()
            return checksum
        except Exception as e:
            logger.error(f"Error calculating checksum for {table_name}: {e}")
            return ""
    
    def get_table_row_count(self, table_name: str, db_connection) -> int:
        """
        Get current row count for table
        
        Args:
            table_name: Name of the table
            db_connection: Database connection
        
        Returns:
            Row count
        """
        try:
            query = text(f"SELECT COUNT(*) as cnt FROM {table_name}")
            result = db_connection.execute(query)
            count = result.scalar()
            return count
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            return 0
    
    def detect_table_changes(self, table_name: str, db_connection) -> Dict[str, Any]:
        """
        Detect if table has changed since last sync
        
        Args:
            table_name: Name of the table
            db_connection: Database connection
        
        Returns:
            Dictionary with change information
        """
        current_checksum = self.get_table_checksum(table_name, db_connection)
        current_count = self.get_table_row_count(table_name, db_connection)
        
        previous_checksum = self.tracking_data["table_checksums"].get(table_name, "")
        previous_count = self.tracking_data["table_row_counts"].get(table_name, 0)
        
        has_changed = (current_checksum != previous_checksum) or (current_count != previous_count)
        
        return {
            "table": table_name,
            "has_changed": has_changed,
            "current_count": current_count,
            "previous_count": previous_count,
            "count_delta": current_count - previous_count,
            "current_checksum": current_checksum
        }
    
    def update_table_tracking(self, table_name: str, checksum: str, row_count: int):
        """
        Update tracking data for a table
        
        Args:
            table_name: Name of the table
            checksum: Current checksum
            row_count: Current row count
        """
        self.tracking_data["table_checksums"][table_name] = checksum
        self.tracking_data["table_row_counts"][table_name] = row_count
        self.tracking_data["table_last_modified"][table_name] = datetime.now().isoformat()
    
    def mark_sync_complete(self):
        """Mark sync as complete and save tracking data"""
        self.tracking_data["last_sync"] = datetime.now().isoformat()
        self._save_tracking_data()
    
    def get_last_sync_time(self) -> str:
        """Get last sync timestamp"""
        return self.tracking_data.get("last_sync", "Never")
