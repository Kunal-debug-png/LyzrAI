"""
Embeddings Processing Module
Handles creation and management of embeddings for graph entities
"""

import os
import json
import hashlib
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from database.neo4j_connector import Neo4jConnector
from config.settings import EmbeddingConfig, SystemConfig

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Creates and manages embeddings for graph entities"""
    
    def __init__(self, neo4j_connector: Neo4jConnector,
                 embedding_config: EmbeddingConfig,
                 system_config: SystemConfig,
                 ontology: Dict[str, Any]):
        """
        Initialize embedding processor
        
        Args:
            neo4j_connector: Neo4j connector instance
            embedding_config: Embedding configuration
            system_config: System configuration
            ontology: Ontology dictionary
        """
        self.neo4j = neo4j_connector
        self.embedding_config = embedding_config
        self.system_config = system_config
        self.ontology = ontology
        
        self.embeddings = CohereEmbeddings(
            cohere_api_key=embedding_config.cohere_api_key,
            model=embedding_config.model
        )
        
        self.vector_store = None
        self.graph_metadata = None
        self.content_hash = None
    
    def extract_graph_data(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Extract all nodes and relationships from Neo4j with their properties
        
        Returns:
            Tuple of (nodes, relationships, schema_info)
        """
        logger.info("\n=== Extracting Graph Data from Neo4j ===")
        
        with self.neo4j.get_session() as session:
            # Extract all nodes with their labels and properties
            nodes_query = """
            MATCH (n)
            RETURN 
                id(n) as node_id,
                labels(n) as labels,
                properties(n) as properties
            """
            nodes_result = session.run(nodes_query)
            nodes = [dict(record) for record in nodes_result]
            logger.info(f"✓ Extracted {len(nodes)} nodes")
            
            # Extract all relationships
            rels_query = """
            MATCH (a)-[r]->(b)
            RETURN 
                id(a) as source_id,
                labels(a)[0] as source_label,
                type(r) as relationship_type,
                properties(r) as rel_properties,
                id(b) as target_id,
                labels(b)[0] as target_label,
                properties(a) as source_props,
                properties(b) as target_props
            """
            rels_result = session.run(rels_query)
            relationships = [dict(record) for record in rels_result]
            logger.info(f"✓ Extracted {len(relationships)} relationships")
            
            # Get schema information
            stats = self.neo4j.get_statistics()
            schema_info = {
                "available": True,
                "node_labels": stats.get("labels", []),
                "relationship_types": stats.get("relationship_types", []),
                "node_counts": stats.get("node_counts", {}),
                "relationship_counts": stats.get("relationship_counts", {})
            }
        
        return nodes, relationships, schema_info
    
    def create_documents_from_nodes(self, nodes: List[Dict]) -> List[Document]:
        """
        Create LangChain documents from graph nodes
        
        Args:
            nodes: List of node dictionaries
        
        Returns:
            List of Document objects
        """
        documents = []
        concept_map = {c['concept']: c for c in self.ontology.get('concepts', [])}
        
        for node in nodes:
            node_id = node['node_id']
            labels = node['labels']
            properties = node['properties']
            
            # Get concept description if available
            concept_desc = ""
            for label in labels:
                if label in concept_map:
                    concept_desc = concept_map[label].get('description', '')
                    break
            
            # Create rich content for embedding
            content = f"Entity: {', '.join(labels)}\n"
            if concept_desc:
                content += f"Description: {concept_desc}\n"
            content += f"Properties: {json.dumps(properties, default=str)}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "node_id": node_id,
                    "labels": labels,
                    "type": "node",
                    "primary_label": labels[0] if labels else "Unknown"
                }
            )
            documents.append(doc)
        
        return documents
    
    def create_documents_from_relationships(self, relationships: List[Dict]) -> List[Document]:
        """
        Create LangChain documents from graph relationships
        
        Args:
            relationships: List of relationship dictionaries
        
        Returns:
            List of Document objects
        """
        documents = []
        
        for rel in relationships:
            source_label = rel['source_label']
            target_label = rel['target_label']
            rel_type = rel['relationship_type']
            
            # Create content describing the relationship
            content = f"Relationship: {source_label} -{rel_type}-> {target_label}\n"
            content += f"Source Properties: {json.dumps(rel['source_props'], default=str)}\n"
            content += f"Target Properties: {json.dumps(rel['target_props'], default=str)}"
            
            if rel['rel_properties']:
                content += f"\nRelationship Properties: {json.dumps(rel['rel_properties'], default=str)}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source_id": rel['source_id'],
                    "target_id": rel['target_id'],
                    "relationship_type": rel_type,
                    "type": "relationship"
                }
            )
            documents.append(doc)
        
        return documents
    
    def calculate_content_hash(self, nodes: List[Dict], relationships: List[Dict]) -> str:
        """
        Calculate hash of graph content for change detection
        
        Args:
            nodes: List of nodes
            relationships: List of relationships
        
        Returns:
            Content hash string
        """
        content_str = json.dumps({
            "nodes": len(nodes),
            "relationships": len(relationships),
            "node_sample": str(nodes[:5]) if nodes else "",
            "rel_sample": str(relationships[:5]) if relationships else ""
        }, sort_keys=True)
        
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents
        
        Args:
            documents: List of documents to embed
        
        Returns:
            FAISS vector store
        """
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info("✓ Vector store created")
        return vector_store
    
    def save_vector_store(self, vector_store: FAISS):
        """Save vector store to disk using FAISS native save"""
        os.makedirs(os.path.dirname(self.system_config.vector_store_path), exist_ok=True)
        
        # Use FAISS's native save method instead of pickle
        # This avoids pickle errors with SimpleQueue objects
        try:
            vector_store.save_local(os.path.dirname(self.system_config.vector_store_path))
            logger.info(f"✓ Vector store saved to {self.system_config.vector_store_path}")
        except Exception as e:
            logger.warning(f"Failed to save with native method, trying pickle: {e}")
            # Fallback to pickle for older versions
            try:
                with open(self.system_config.vector_store_path, 'wb') as f:
                    pickle.dump(vector_store, f)
                logger.info(f"✓ Vector store saved to {self.system_config.vector_store_path}")
            except Exception as e2:
                logger.error(f"Failed to save vector store: {e2}")
                raise
    
    def load_vector_store(self) -> Optional[FAISS]:
        """Load vector store from disk using FAISS native load"""
        vector_store_dir = os.path.dirname(self.system_config.vector_store_path)
        
        # Try FAISS native load first
        if os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
            try:
                vector_store = FAISS.load_local(
                    vector_store_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✓ Vector store loaded from {vector_store_dir}")
                return vector_store
            except Exception as e:
                logger.warning(f"Failed to load with native method: {e}")
        
        # Fallback to pickle for older saved stores
        if os.path.exists(self.system_config.vector_store_path):
            try:
                with open(self.system_config.vector_store_path, 'rb') as f:
                    vector_store = pickle.load(f)
                logger.info(f"✓ Vector store loaded from {self.system_config.vector_store_path}")
                return vector_store
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")
        
        return None
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save graph metadata to disk"""
        os.makedirs(os.path.dirname(self.system_config.graph_metadata_path), exist_ok=True)
        with open(self.system_config.graph_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✓ Metadata saved to {self.system_config.graph_metadata_path}")
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load graph metadata from disk"""
        if os.path.exists(self.system_config.graph_metadata_path):
            try:
                with open(self.system_config.graph_metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✓ Metadata loaded from {self.system_config.graph_metadata_path}")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return None
    
    def process_graph(self, force_reprocess: bool = False) -> Tuple[FAISS, Dict[str, Any], str]:
        """
        Process graph and create embeddings
        
        Args:
            force_reprocess: Force reprocessing even if cache exists
        
        Returns:
            Tuple of (vector_store, metadata, content_hash)
        """
        # Extract graph data
        nodes, relationships, schema_info = self.extract_graph_data()
        
        # Calculate content hash
        content_hash = self.calculate_content_hash(nodes, relationships)
        
        # Check if we can use cached data
        if not force_reprocess:
            cached_metadata = self.load_metadata()
            if cached_metadata and cached_metadata.get('content_hash') == content_hash:
                logger.info("✓ Graph unchanged, using cached embeddings")
                vector_store = self.load_vector_store()
                if vector_store:
                    return vector_store, cached_metadata, content_hash
        
        logger.info("Processing graph and creating new embeddings...")
        
        # Create documents
        node_docs = self.create_documents_from_nodes(nodes)
        rel_docs = self.create_documents_from_relationships(relationships)
        all_docs = node_docs + rel_docs
        
        logger.info(f"Created {len(node_docs)} node documents and {len(rel_docs)} relationship documents")
        
        # Create vector store
        vector_store = self.create_vector_store(all_docs)
        
        # Prepare metadata
        metadata = {
            "content_hash": content_hash,
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "total_documents": len(all_docs),
            "schema_info": schema_info,
            "ontology": self.ontology
        }
        
        # Save to disk
        self.save_vector_store(vector_store)
        self.save_metadata(metadata)
        
        self.vector_store = vector_store
        self.graph_metadata = metadata
        self.content_hash = content_hash
        
        return vector_store, metadata, content_hash
    
    def update_embeddings_for_tables(self, table_names: List[str]) -> FAISS:
        """
        Update embeddings for specific tables (incremental update)
        
        Args:
            table_names: List of table names that changed
        
        Returns:
            Updated vector store
        """
        logger.info(f"\n=== Updating Embeddings for Changed Tables ===")
        
        concept_map = {c['table']: c for c in self.ontology.get('concepts', [])}
        
        # Extract nodes from changed tables
        documents = []
        
        with self.neo4j.get_session() as session:
            for table_name in table_names:
                if table_name not in concept_map:
                    continue
                
                concept = concept_map[table_name]
                concept_name = concept['concept']
                
                if concept.get('is_junction', False):
                    continue
                
                # Get all nodes of this type
                query = f"""
                MATCH (n:{concept_name})
                RETURN id(n) as node_id, labels(n) as labels, properties(n) as properties
                """
                result = session.run(query)
                
                for record in result:
                    node_id = record["node_id"]
                    labels = record["labels"]
                    props = record["properties"]
                    
                    # Create document content
                    content = f"Entity: {', '.join(labels)}\n"
                    content += f"Properties: {json.dumps(props, default=str)}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "node_id": node_id,
                            "labels": labels,
                            "type": "node",
                            "concept": concept_name
                        }
                    )
                    documents.append(doc)
        
        if not documents:
            logger.info("  No documents to update")
            return self.vector_store
        
        logger.info(f"  Creating embeddings for {len(documents)} entities...")
        
        # Load existing vector store or create new
        vector_store = self.load_vector_store()
        
        if vector_store is None:
            vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add new documents to existing store
            new_store = FAISS.from_documents(documents, self.embeddings)
            vector_store.merge_from(new_store)
        
        self.save_vector_store(vector_store)
        logger.info(f"  ✓ Updated embeddings for {len(documents)} entities")
        
        return vector_store
