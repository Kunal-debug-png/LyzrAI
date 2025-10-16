import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RDBMSGraphEmbeddingProcessor:
    """
    Creates embeddings for RDBMS-based knowledge graph entities and relationships.
    Enables semantic search over graph nodes and their properties.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 cohere_api_key: str, ontology_path: str = "ontology_output.json"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.cohere_api_key = cohere_api_key
        self.embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0"
        )
        
        # Load ontology for semantic context
        self.ontology = self._load_ontology(ontology_path)
        
        self.vector_store = None
        self.graph_metadata = None
        self.content_hash = None
        
    def _load_ontology(self, path: str) -> Dict:
        """Load the generated ontology for semantic understanding"""
        try:
            with open(path, 'r') as f:
                ontology = json.load(f)
            logger.info(f"✓ Loaded ontology: {ontology.get('domain', 'Unknown')}")
            return ontology
        except Exception as e:
            logger.warning(f"Could not load ontology: {e}")
            return {}
    
    def close(self):
        self.driver.close()
    
    def extract_graph_data(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Extract all nodes and relationships from Neo4j with their properties.
        Returns: (nodes, relationships, schema_info)
        """
        logger.info("\n=== Extracting Graph Data from Neo4j ===")
        
        with self.driver.session() as session:
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
            schema_query = """
            CALL db.schema.visualization()
            """
            try:
                schema_result = session.run(schema_query)
                schema_info = {"available": True}
            except:
                schema_info = {"available": False}
            
            # Get node label counts
            label_counts_query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
            """
            label_result = session.run(label_counts_query)
            schema_info["label_counts"] = {record["label"]: record["count"] 
                                          for record in label_result}
            
            # Get relationship type counts
            rel_counts_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """
            rel_result = session.run(rel_counts_query)
            schema_info["relationship_counts"] = {record["type"]: record["count"] 
                                                 for record in rel_result}
        
        return nodes, relationships, schema_info
    
    def create_semantic_documents(self, nodes: List[Dict], 
                                  relationships: List[Dict]) -> List[Document]:
        """
        Create rich semantic documents from graph data for embedding.
        Each document represents either a node or a relationship with context.
        """
        logger.info("\n=== Creating Semantic Documents ===")
        documents = []
        
        # Create node documents with rich context
        for node in nodes:
            node_id = node['node_id']
            labels = node['labels']
            props = node['properties']
            
            # Get concept description from ontology
            label = labels[0] if labels else "Unknown"
            concept_info = self._get_concept_info(label)
            
            # Build semantic text representation
            text_parts = [
                f"Entity: {label}",
                f"Description: {concept_info.get('description', f'A {label} entity')}",
            ]
            
            # Add properties in natural language
            if props:
                prop_texts = []
                for key, value in props.items():
                    # Get semantic property name
                    semantic_name = self._get_property_semantic_name(label, key)
                    prop_texts.append(f"{semantic_name}: {value}")
                text_parts.append("Properties: " + ", ".join(prop_texts))
            
            text = " | ".join(text_parts)
            
            metadata = {
                "type": "node",
                "node_id": node_id,
                "labels": labels,
                "properties": props,
                "concept": label
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        # Create relationship documents with context
        for rel in relationships:
            source_label = rel['source_label']
            target_label = rel['target_label']
            rel_type = rel['relationship_type']
            
            # Get relationship description from ontology
            rel_info = self._get_relationship_info(source_label, target_label, rel_type)
            
            # Build semantic text representation
            source_desc = self._format_entity_brief(rel['source_props'], source_label)
            target_desc = self._format_entity_brief(rel['target_props'], target_label)
            
            text = (
                f"Relationship: {source_label} {rel_type} {target_label} | "
                f"Description: {rel_info.get('description', f'{source_label} relates to {target_label}')} | "
                f"Source: {source_desc} | "
                f"Target: {target_desc}"
            )
            
            metadata = {
                "type": "relationship",
                "source_id": rel['source_id'],
                "target_id": rel['target_id'],
                "source_label": source_label,
                "target_label": target_label,
                "relationship_type": rel_type,
                "properties": rel['rel_properties']
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        logger.info(f"✓ Created {len(documents)} semantic documents")
        return documents
    
    def _get_concept_info(self, label: str) -> Dict:
        """Get semantic information about a concept from ontology"""
        if not self.ontology or 'concepts' not in self.ontology:
            return {}
        
        for concept in self.ontology['concepts']:
            if concept.get('concept') == label:
                return concept
        return {}
    
    def _get_property_semantic_name(self, label: str, prop_name: str) -> str:
        """Get semantic name for a property"""
        concept_info = self._get_concept_info(label)
        if concept_info and 'properties' in concept_info:
            for prop in concept_info['properties']:
                if prop['column'] == prop_name:
                    return prop.get('semantic_name', prop_name)
        return prop_name
    
    def _get_relationship_info(self, source: str, target: str, rel_type: str) -> Dict:
        """Get semantic information about a relationship from ontology"""
        if not self.ontology or 'relationships' not in self.ontology:
            return {}
        
        for rel in self.ontology['relationships']:
            if (rel.get('relationship_type') == rel_type or
                (self._get_concept_by_table(rel.get('from_table')) == source and
                 self._get_concept_by_table(rel.get('to_table')) == target)):
                return rel
        return {}
    
    def _get_concept_by_table(self, table_name: str) -> Optional[str]:
        """Get concept name from table name"""
        if not self.ontology or 'concepts' not in self.ontology:
            return None
        
        for concept in self.ontology['concepts']:
            if concept.get('table') == table_name:
                return concept.get('concept')
        return None
    
    def _format_entity_brief(self, props: Dict, label: str) -> str:
        """Create a brief description of an entity from its properties"""
        if not props:
            return label
        
        # Try to find identifying properties (id, name, title, etc.)
        identifying_keys = ['name', 'title', 'id', 'email', 'username']
        for key in identifying_keys:
            if key in props:
                return f"{label} ({props[key]})"
        
        # Return first property
        first_key = next(iter(props))
        return f"{label} ({props[first_key]})"
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        logger.info("\n=== Creating FAISS Vector Store ===")
        
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"✓ Created vector store with {len(documents)} documents")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def generate_content_hash(self, nodes: List[Dict], relationships: List[Dict]) -> str:
        """Generate hash for graph content versioning"""
        content_str = json.dumps({
            "nodes_count": len(nodes),
            "rels_count": len(relationships),
            "ontology_domain": self.ontology.get('domain', 'unknown')
        }, sort_keys=True)
        
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def save_artifacts(self, vector_store: FAISS, graph_metadata: Dict, 
                      output_dir: str = "./graph_artifacts"):
        """Save vector store and metadata to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        vector_store_path = os.path.join(output_dir, "faiss_index")
        vector_store.save_local(vector_store_path)
        logger.info(f"✓ Saved FAISS index to {vector_store_path}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "graph_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(graph_metadata, f)
        logger.info(f"✓ Saved metadata to {metadata_path}")
        
        # Save content hash
        hash_path = os.path.join(output_dir, "content_hash.txt")
        with open(hash_path, 'w') as f:
            f.write(self.content_hash)
        logger.info(f"✓ Saved content hash: {self.content_hash}")
    
    def load_artifacts(self, output_dir: str = "./graph_artifacts") -> Tuple[FAISS, Dict]:
        """Load previously saved artifacts"""
        try:
            vector_store_path = os.path.join(output_dir, "faiss_index")
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            metadata_path = os.path.join(output_dir, "graph_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                self.graph_metadata = pickle.load(f)
            
            hash_path = os.path.join(output_dir, "content_hash.txt")
            with open(hash_path, 'r') as f:
                self.content_hash = f.read().strip()
            
            logger.info("✓ Loaded artifacts successfully")
            return self.vector_store, self.graph_metadata
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            return None, None
    
    def process_graph(self, save_artifacts: bool = True) -> Tuple[FAISS, Dict, str]:
        """
        Main processing pipeline:
        1. Extract graph data from Neo4j
        2. Create semantic documents
        3. Generate embeddings and vector store
        4. Save artifacts
        
        Returns: (vector_store, graph_metadata, content_hash)
        """
        logger.info("\n" + "=" * 60)
        logger.info("RDBMS GRAPH EMBEDDING PROCESSOR")
        logger.info("=" * 60)
        
        # Step 1: Extract graph data
        nodes, relationships, schema_info = self.extract_graph_data()
        
        # Step 2: Create semantic documents
        documents = self.create_semantic_documents(nodes, relationships)
        
        # Step 3: Create vector store
        self.vector_store = self.create_vector_store(documents)
        
        # Step 4: Prepare metadata
        self.graph_metadata = {
            "nodes": nodes,
            "relationships": relationships,
            "schema_info": schema_info,
            "ontology": self.ontology,
            "documents": documents
        }
        
        # Step 5: Generate content hash
        self.content_hash = self.generate_content_hash(nodes, relationships)
        
        # Step 6: Save artifacts
        if save_artifacts:
            self.save_artifacts(self.vector_store, self.graph_metadata)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ GRAPH PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"📊 Nodes: {len(nodes)}")
        logger.info(f"🔗 Relationships: {len(relationships)}")
        logger.info(f"📝 Documents: {len(documents)}")
        logger.info(f"🔑 Content Hash: {self.content_hash}")
        logger.info("=" * 60 + "\n")
        
        return self.vector_store, self.graph_metadata, self.content_hash


def process_rdbms_graph_sync(neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                             cohere_api_key: str, ontology_path: str = "ontology_output.json",
                             force_reprocess: bool = False) -> Tuple[FAISS, Dict, str]:
    """
    Synchronous wrapper for processing RDBMS graph.
    
    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        cohere_api_key: Cohere API key for embeddings
        ontology_path: Path to ontology JSON file
        force_reprocess: Force reprocessing even if artifacts exist
        
    Returns:
        (vector_store, graph_metadata, content_hash)
    """
    processor = RDBMSGraphEmbeddingProcessor(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        cohere_api_key=cohere_api_key,
        ontology_path=ontology_path
    )
    
    try:
        # Try to load existing artifacts if not forcing reprocess
        if not force_reprocess:
            logger.info("Checking for existing artifacts...")
            vector_store, graph_metadata = processor.load_artifacts()
            if vector_store and graph_metadata:
                logger.info("✓ Using cached artifacts")
                return vector_store, graph_metadata, processor.content_hash
        
        # Process graph fresh
        result = processor.process_graph(save_artifacts=True)
        return result
    finally:
        processor.close()