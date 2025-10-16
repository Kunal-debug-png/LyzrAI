"""
RDBMS Database Connector Module
Handles connections and operations for relational databases
"""

from typing import Generator, Optional
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from config.settings import RDBMSConfig
import logging

logger = logging.getLogger(__name__)


class RDBMSConnector:
    """Manages RDBMS database connections and operations"""
    
    def __init__(self, config: RDBMSConfig):
        """
        Initialize RDBMS connector
        
        Args:
            config: RDBMS configuration object
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._inspector = None
    
    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine"""
        if self._engine is None:
            self._engine = create_engine(
                self.config.connection_string,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True  # Verify connections before using
            )
            logger.info("RDBMS engine created")
        return self._engine
    
    @property
    def inspector(self):
        """Get SQLAlchemy inspector for schema introspection"""
        if self._inspector is None:
            self._inspector = inspect(self.engine)
        return self._inspector
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                autoflush=False,
                autocommit=False,
                bind=self.engine
            )
        return self._session_factory
    
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session (context manager)
        
        Yields:
            SQLAlchemy session
        """
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()
    
    def get_table_names(self) -> list[str]:
        """Get all table names in the database"""
        return self.inspector.get_table_names()
    
    def get_columns(self, table_name: str) -> list[dict]:
        """Get column information for a table"""
        return self.inspector.get_columns(table_name)
    
    def get_primary_keys(self, table_name: str) -> list[str]:
        """Get primary key columns for a table"""
        pk_constraint = self.inspector.get_pk_constraint(table_name)
        return pk_constraint.get("constrained_columns", []) if pk_constraint else []
    
    def get_foreign_keys(self, table_name: str) -> list[dict]:
        """Get foreign key information for a table"""
        return self.inspector.get_foreign_keys(table_name)
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("RDBMS connection test successful")
            return True
        except Exception as e:
            logger.error(f"RDBMS connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            logger.info("RDBMS engine disposed")
            self._engine = None
            self._session_factory = None
            self._inspector = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Singleton instance for backward compatibility
_default_connector: Optional[RDBMSConnector] = None


def get_rdbms_connector(config: Optional[RDBMSConfig] = None) -> RDBMSConnector:
    """
    Get or create default RDBMS connector
    
    Args:
        config: Optional config, uses default if not provided
    
    Returns:
        RDBMSConnector instance
    """
    global _default_connector
    
    if _default_connector is None:
        if config is None:
            config = RDBMSConfig.from_env()
        _default_connector = RDBMSConnector(config)
    
    return _default_connector


def get_db() -> Generator[Session, None, None]:
    """
    Legacy function for backward compatibility
    Get a database session
    """
    connector = get_rdbms_connector()
    yield from connector.get_session()


# Legacy exports for backward compatibility
def get_engine() -> Engine:
    """Get the default engine (legacy)"""
    return get_rdbms_connector().engine


engine = None  # Will be set lazily


def __getattr__(name):
    """Lazy loading for backward compatibility"""
    global engine
    if name == "engine":
        if engine is None:
            engine = get_engine()
        return engine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
