"""
Legacy RDBMS Connector - Backward Compatibility Wrapper
This module provides backward compatibility with the old connector interface.
New code should use database.rdbms_connector instead.
"""

from database.rdbms_connector import get_db, get_engine

# Legacy exports
engine = None  # Will be lazily loaded
Session = None  # Will be lazily loaded


def __getattr__(name):
    """Lazy loading for backward compatibility"""
    global engine, Session
    
    if name == "engine":
        if engine is None:
            engine = get_engine()
        return engine
    elif name == "Session":
        if Session is None:
            from database.rdbms_connector import get_rdbms_connector
            connector = get_rdbms_connector()
            Session = connector.session_factory
        return Session
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")