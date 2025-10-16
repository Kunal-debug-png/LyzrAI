import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator
from enum import Enum
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    VECTOR_SEARCH = "vector_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    LOGICAL_FILTER = "logical_filter"
    CYPHER_QUERY = "cypher_query"
    HYBRID = "hybrid"


class AgenticRDBMSGraphQA:
    """
    Agentic Q&A system for RDBMS-based knowledge graphs.
    Combines vector search, graph traversal, and logical filtering with autonomous agent reasoning.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 groq_api_key: str, vector_store=None, graph_metadata: Dict = None):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            streaming=True
        )
        self.vector_store = vector_store
        self.graph_metadata = graph_metadata
        self.ontology = graph_metadata.get('ontology', {}) if graph_metadata else {}
        
        # Reasoning chain for transparency
        self.reasoning_chain = []
    
    def close(self):
        self.driver.close()
    
    def _add_reasoning_step(self, step: str, details: Dict = None):
        """Add a step to the reasoning chain for transparency"""
        self.reasoning_chain.append({
            "step": step,
            "details": details or {}
        })
        logger.info(f"🧠 Reasoning: {step}")
    
    def analyze_query(self, question: str) -> Dict[str, Any]:
        """
        Use LLM to analyze the query and determine optimal retrieval strategy.
        This is the autonomous decision-making agent.
        """
        self._add_reasoning_step("Analyzing query intent and complexity")
        
        # Get schema context
        schema_summary = self._get_schema_summary()
        
        analysis_prompt = f"""You are an intelligent query analyzer for a knowledge graph database.

KNOWLEDGE GRAPH SCHEMA:
{schema_summary}

AVAILABLE RETRIEVAL STRATEGIES:
1. vector_search: Semantic similarity search (best for: "find similar", "what is related to", conceptual questions)
2. graph_traversal: Multi-hop path finding (best for: "how are X and Y connected", "path between", relationship chains)
3. logical_filter: Attribute-based filtering (best for: "all X where", "filter by", specific property constraints)
4. cypher_query: Direct database query (best for: complex aggregations, counts, statistical queries)
5. hybrid: Combination of above (best for: complex multi-faceted queries)

USER QUESTION: {question}

Analyze this query and respond with a JSON object:
{{
  "query_type": "relationship|factual|statistical|exploratory|complex",
  "primary_strategy": "vector_search|graph_traversal|logical_filter|cypher_query|hybrid",
  "secondary_strategies": ["strategy1", "strategy2"],
  "entities_mentioned": ["entity1", "entity2"],
  "relationship_focus": true/false,
  "requires_aggregation": true/false,
  "complexity_score": 1-5,
  "reasoning": "why this strategy is best"
}}

Return ONLY the JSON object, no other text."""

        try:
            response = self.llm.invoke(analysis_prompt)
            content = response.content.strip()
            
            # Extract JSON from markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            self._add_reasoning_step("Query analysis complete", analysis)
            return analysis
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}, using default strategy")
            return {
                "query_type": "exploratory",
                "primary_strategy": "hybrid",
                "complexity_score": 3,
                "reasoning": "Default fallback strategy"
            }
    
    def _get_schema_summary(self) -> str:
        """Generate a concise schema summary for the LLM"""
        if not self.graph_metadata:
            return "Schema not available"
        
        schema_info = self.graph_metadata.get('schema_info', {})
        ontology = self.ontology
        
        summary = []
        
        # Domain context
        if ontology.get('domain'):
            summary.append(f"Domain: {ontology['domain']}")
        
        # Node types
        label_counts = schema_info.get('label_counts', {})
        if label_counts:
            summary.append("\nEntity Types:")
            for label, count in list(label_counts.items())[:10]:
                concept_info = self._get_concept_description(label)
                summary.append(f"  - {label}: {count} instances ({concept_info})")
        
        # Relationship types
        rel_counts = schema_info.get('relationship_counts', {})
        if rel_counts:
            summary.append("\nRelationship Types:")
            for rel_type, count in list(rel_counts.items())[:10]:
                rel_info = self._get_relationship_description(rel_type)
                summary.append(f"  - {rel_type}: {count} connections ({rel_info})")
        
        return "\n".join(summary)
    
    def _get_concept_description(self, label: str) -> str:
        """Get description of a concept from ontology"""
        if not self.ontology or 'concepts' not in self.ontology:
            return "entity"
        
        for concept in self.ontology['concepts']:
            if concept.get('concept') == label:
                return concept.get('description', 'entity')
        return "entity"
    
    def _get_relationship_description(self, rel_type: str) -> str:
        """Get description of a relationship from ontology"""
        if not self.ontology or 'relationships' not in self.ontology:
            return "relates to"
        
        for rel in self.ontology['relationships']:
            if rel.get('relationship_type') == rel_type:
                return rel.get('description', 'relates to')
        return "relates to"
    
    def vector_search(self, question: str, top_k: int = 5) -> List[Dict]:
        """Semantic similarity search using embeddings"""
        self._add_reasoning_step(f"Executing vector search (top_k={top_k})")
        
        if not self.vector_store:
            logger.warning("Vector store not available")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(question, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "type": doc.metadata.get('type', 'unknown')
                })
            
            self._add_reasoning_step(f"Found {len(formatted_results)} relevant items", 
                                    {"top_score": formatted_results[0]["similarity_score"] if formatted_results else 0})
            return formatted_results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def graph_traversal(self, start_entities: List[str], max_depth: int = 3) -> List[Dict]:
        """Multi-hop graph traversal to find relationship paths"""
        self._add_reasoning_step(f"Executing graph traversal (depth={max_depth})", 
                                {"start_entities": start_entities})
        
        paths = []
        
        with self.driver.session() as session:
            for entity in start_entities:
                # Find paths from this entity
                query = f"""
                MATCH path = (start)-[*1..{max_depth}]-(end)
                WHERE ANY(label IN labels(start) WHERE label CONTAINS $entity)
                   OR ANY(prop IN keys(start) WHERE toString(start[prop]) CONTAINS $entity)
                RETURN path
                LIMIT 20
                """
                
                try:
                    result = session.run(query, entity=entity)
                    for record in result:
                        path = record['path']
                        paths.append(self._format_path(path))
                except Exception as e:
                    logger.error(f"Graph traversal error for {entity}: {e}")
        
        self._add_reasoning_step(f"Found {len(paths)} relationship paths")
        return paths
    
    def logical_filter(self, filters: Dict[str, Any]) -> List[Dict]:
        """Property-based filtering on nodes"""
        self._add_reasoning_step("Executing logical filter", {"filters": filters})
        
        results = []
        
        with self.driver.session() as session:
            # Build dynamic WHERE clause
            where_clauses = []
            params = {}
            
            for key, value in filters.items():
                param_name = f"param_{key}"
                if isinstance(value, str):
                    where_clauses.append(f"ANY(prop IN keys(n) WHERE n[prop] =~ $"  + param_name + ")")
                    params[param_name] = f"(?i).*{value}.*"  # Case-insensitive regex
                else:
                    where_clauses.append(f"ANY(prop IN keys(n) WHERE n[prop] = ${param_name})")
                    params[param_name] = value
            
            where_str = " AND ".join(where_clauses) if where_clauses else "true"
            
            query = f"""
            MATCH (n)
            WHERE {where_str}
            RETURN n, labels(n) as labels
            LIMIT 50
            """
            
            try:
                result = session.run(query, **params)
                for record in result:
                    node = record['n']
                    results.append({
                        "labels": record['labels'],
                        "properties": dict(node)
                    })
            except Exception as e:
                logger.error(f"Logical filter error: {e}")
        
        self._add_reasoning_step(f"Filter matched {len(results)} entities")
        return results
    
    def generate_cypher_query(self, question: str) -> Optional[str]:
        """Use LLM to generate Cypher query from natural language"""
        self._add_reasoning_step("Generating Cypher query from natural language")
        
        schema_summary = self._get_schema_summary()
        
        # Extract sample properties from graph metadata
        sample_properties = self._get_sample_properties()
        
        cypher_prompt = f"""You are a Cypher query expert. Generate a Cypher query for Neo4j.

GRAPH SCHEMA:
{schema_summary}

SAMPLE PROPERTIES:
{sample_properties}

USER QUESTION: {question}

Generate a Cypher query that answers this question. Follow these rules:
1. Use MATCH, WHERE, RETURN appropriately
2. For names/text search, use case-insensitive matching: WHERE toLower(n.name) CONTAINS toLower($name)
3. Return ALL relevant properties, not just counts
4. Limit results to 50 unless doing aggregation (COUNT, SUM, AVG, etc.)
5. When listing entities, return their properties, not just counts
6. Use proper property matching based on the sample properties shown above

Examples:
- "list all customers" -> MATCH (n:Customer) RETURN n.customer_id, n.email, n.registration_date LIMIT 50
- "who is John" -> MATCH (n) WHERE toLower(n.name) CONTAINS 'john' OR toLower(n.first_name) CONTAINS 'john' RETURN n, labels(n) LIMIT 10
- "how many orders" -> MATCH (n:CustomerOrder) RETURN COUNT(n) as total_orders

Return ONLY the Cypher query, no explanations or markdown."""

        try:
            response = self.llm.invoke(cypher_prompt)
            cypher = response.content.strip()
            
            # Clean up markdown if present
            if "```cypher" in cypher:
                cypher = cypher.split("```cypher")[1].split("```")[0].strip()
            elif "```" in cypher:
                cypher = cypher.split("```")[1].split("```")[0].strip()
            
            self._add_reasoning_step("Cypher query generated", {"query": cypher[:200]})
            return cypher
        except Exception as e:
            logger.error(f"Cypher generation error: {e}")
            return None
    
    def _get_sample_properties(self) -> str:
        """Get sample properties from each node type"""
        if not self.graph_metadata or 'nodes' not in self.graph_metadata:
            return "No sample properties available"
        
        nodes = self.graph_metadata['nodes']
        samples_by_label = {}
        
        # Group by label and get first example
        for node in nodes[:30]:  # Look at first 30 nodes
            labels = node.get('labels', [])
            if labels:
                label = labels[0]
                if label not in samples_by_label:
                    props = node.get('properties', {})
                    if props:
                        samples_by_label[label] = props
        
        # Format for LLM
        result = []
        for label, props in list(samples_by_label.items())[:10]:
            prop_list = ", ".join([f"{k}: {type(v).__name__}" for k, v in list(props.items())[:5]])
            result.append(f"  {label}: {prop_list}")
        
        return "\n".join(result) if result else "No sample properties available"
    
    def execute_cypher(self, cypher: str) -> List[Dict]:
        """Execute a Cypher query and return results"""
        self._add_reasoning_step("Executing Cypher query")
        
        results = []
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher)
                for record in result:
                    results.append(dict(record))
            except Exception as e:
                logger.error(f"Cypher execution error: {e}")
                self._add_reasoning_step("Cypher execution failed", {"error": str(e)})
        
        self._add_reasoning_step(f"Cypher returned {len(results)} results")
        return results
    
    def _format_path(self, path) -> Dict:
        """Format a Neo4j path object"""
        nodes = []
        relationships = []
        
        for node in path.nodes:
            nodes.append({
                "labels": list(node.labels),
                "properties": dict(node)
            })
        
        for rel in path.relationships:
            relationships.append({
                "type": rel.type,
                "properties": dict(rel)
            })
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }
    
    def hybrid_retrieval(self, question: str, analysis: Dict) -> Dict[str, Any]:
        """
        Execute hybrid retrieval combining multiple strategies.
        This is where the agent orchestrates different retrieval methods.
        """
        self._add_reasoning_step("Executing hybrid retrieval strategy")
        
        results = {
            "vector_results": [],
            "graph_paths": [],
            "filtered_entities": [],
            "cypher_results": []
        }
        
        # Always do vector search first for context
        results['vector_results'] = self.vector_search(question, top_k=10)
        
        # Extract entities from question for traversal
        entities = self._extract_entities_from_question(question)
        if entities:
            results['graph_paths'] = self.graph_traversal(entities, max_depth=2)
        
        # Try Cypher generation for most queries
        cypher = self.generate_cypher_query(question)
        if cypher:
            results['cypher_results'] = self.execute_cypher(cypher)
        
        return results
    
    def _extract_entities_from_question(self, question: str) -> List[str]:
        """Extract potential entity names from question"""
        # Simple extraction - look for capitalized words and common patterns
        import re
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', question)
        quoted.extend(re.findall(r"'([^']+)'", question))
        
        # Extract capitalized words (potential names)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        
        # Common entity patterns
        patterns = [
            r'customer[s]?\s+(?:named\s+)?(\w+(?:\s+\w+)?)',
            r'user[s]?\s+(?:named\s+)?(\w+(?:\s+\w+)?)',
            r'product[s]?\s+(?:named\s+)?(\w+(?:\s+\w+)?)',
            r'order[s]?\s+(?:for\s+)?(\w+(?:\s+\w+)?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            quoted.extend(matches)
        
        return list(set(quoted + capitalized))
    
    def synthesize_answer(self, question: str, retrieval_results: Dict, 
                         analysis: Dict) -> Generator[str, None, None]:
        """
        Use LLM to synthesize final answer from all retrieval results.
        Streams the response with reasoning chain.
        """
        self._add_reasoning_step("Synthesizing final answer from all sources")
        
        # Prepare context from all retrieval sources
        context_parts = []
        has_data = False
        
        # Add Cypher results FIRST (most accurate)
        if retrieval_results.get('cypher_results'):
            has_data = True
            context_parts.append("=== DATABASE QUERY RESULTS ===")
            cypher_results = retrieval_results['cypher_results']
            
            if len(cypher_results) > 0:
                context_parts.append(f"Found {len(cypher_results)} result(s):\n")
                for i, result in enumerate(cypher_results[:30], 1):  # Show up to 30 results
                    # Format result nicely
                    formatted = self._format_cypher_result(result)
                    context_parts.append(f"{i}. {formatted}")
            else:
                context_parts.append("No results found from database query.")
        
        # Add vector search results
        if retrieval_results.get('vector_results'):
            has_data = True
            context_parts.append("\n=== SEMANTIC SEARCH CONTEXT ===")
            for i, result in enumerate(retrieval_results['vector_results'][:5], 1):
                context_parts.append(f"{i}. {result['content']}")
                if result['metadata'].get('type') == 'node':
                    props = result['metadata'].get('properties', {})
                    if props:
                        prop_str = ", ".join([f"{k}={v}" for k, v in list(props.items())[:3]])
                        context_parts.append(f"   Properties: {prop_str}")
        
        # Add graph paths
        if retrieval_results.get('graph_paths'):
            has_data = True
            context_parts.append("\n=== RELATIONSHIP PATHS ===")
            for i, path in enumerate(retrieval_results['graph_paths'][:5], 1):
                path_str = self._format_path_for_llm(path)
                context_parts.append(f"{i}. {path_str}")
        
        if not has_data:
            context_parts.append("No data retrieved from any source.")
        
        context = "\n".join(context_parts)
        
        synthesis_prompt = f"""You are analyzing a knowledge graph database about: {self.ontology.get('domain', 'an e-commerce platform')}.

RETRIEVED DATA:
{context}

USER QUESTION: {question}

Based on the retrieved data above, provide a direct, factual answer.

IMPORTANT RULES:
1. If DATABASE QUERY RESULTS are present, USE THEM as your PRIMARY source - they are the actual data
2. Answer with SPECIFIC data from the results (names, numbers, properties)
3. If listing items, list them clearly with their details
4. If the data shows specific values, state them explicitly
5. Be concise and direct - don't apologize or explain limitations unless truly no data found
6. Use SEMANTIC SEARCH CONTEXT for additional understanding only
7. If no relevant data found, say so clearly in one sentence

Answer:"""

        try:
            # Stream the response
            for chunk in self.llm.stream(synthesis_prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n\n❌ Error generating answer: {str(e)}"
    
    def _format_cypher_result(self, result: Dict) -> str:
        """Format a single Cypher result for better readability"""
        parts = []
        for key, value in result.items():
            if hasattr(value, '__dict__'):  # Neo4j node object
                # It's a node
                try:
                    labels = list(value.labels) if hasattr(value, 'labels') else []
                    props = dict(value) if hasattr(value, '__iter__') else {}
                    label_str = labels[0] if labels else "Entity"
                    prop_str = ", ".join([f"{k}={v}" for k, v in list(props.items())[:5]])
                    parts.append(f"{label_str}({prop_str})")
                except:
                    parts.append(f"{key}={value}")
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # List of items
                parts.append(f"{key}=[{', '.join(str(v) for v in value[:5])}]")
            else:
                # Simple value
                parts.append(f"{key}={value}")
        
        return " | ".join(parts) if parts else str(result)
    
    def _format_path_for_llm(self, path: Dict) -> str:
        """Format a path into readable text for LLM"""
        parts = []
        nodes = path.get('nodes', [])
        rels = path.get('relationships', [])
        
        for i, node in enumerate(nodes):
            label = node['labels'][0] if node['labels'] else 'Entity'
            # Get identifying property
            props = node['properties']
            identifier = self._get_entity_identifier(props)
            parts.append(f"{label}({identifier})")
            
            if i < len(rels):
                rel = rels[i]
                parts.append(f"-[{rel['type']}]->")
        
        return " ".join(parts)
    
    def _get_entity_identifier(self, props: Dict) -> str:
        """Get identifying property from entity"""
        identifying_keys = ['name', 'title', 'id', 'email', 'username']
        for key in identifying_keys:
            if key in props:
                return str(props[key])
        
        # Return first property
        if props:
            first_key = next(iter(props))
            return str(props[first_key])
        return "unknown"
    
    def ask(self, question: str, stream: bool = True) -> Generator[str, None, None] or str:
        """
        Main entry point for asking questions.
        Autonomous agent decides best retrieval strategy and executes it.
        
        Args:
            question: Natural language question
            stream: Whether to stream the response
            
        Returns:
            Generator yielding response chunks if streaming, otherwise full response
        """
        # Reset reasoning chain
        self.reasoning_chain = []
        
        logger.info("\n" + "=" * 60)
        logger.info(f"❓ Question: {question}")
        logger.info("=" * 60)
        
        # Step 1: Analyze query
        analysis = self.analyze_query(question)
        
        # Step 2: Execute retrieval based on analysis - ALWAYS get results
        primary_strategy = analysis.get('primary_strategy', 'hybrid')
        retrieval_results = {}
        
        if primary_strategy == 'vector_search':
            retrieval_results = {
                'vector_results': self.vector_search(question, top_k=10),
                'cypher_results': []
            }
            # Also try cypher
            cypher = self.generate_cypher_query(question)
            if cypher:
                retrieval_results['cypher_results'] = self.execute_cypher(cypher)
                
        elif primary_strategy == 'graph_traversal':
            entities = self._extract_entities_from_question(question)
            retrieval_results = {
                'graph_paths': self.graph_traversal(entities if entities else [question], max_depth=3),
                'vector_results': self.vector_search(question, top_k=5)
            }
            
        elif primary_strategy == 'logical_filter':
            # Execute vector search + cypher
            retrieval_results = {
                'vector_results': self.vector_search(question, top_k=10),
                'cypher_results': []
            }
            cypher = self.generate_cypher_query(question)
            if cypher:
                retrieval_results['cypher_results'] = self.execute_cypher(cypher)
                
        elif primary_strategy == 'cypher_query':
            cypher = self.generate_cypher_query(question)
            retrieval_results = {
                'cypher_results': self.execute_cypher(cypher) if cypher else [],
                'vector_results': self.vector_search(question, top_k=5)
            }
            
        else:  # hybrid - get EVERYTHING
            retrieval_results = self.hybrid_retrieval(question, analysis)
        
        # Step 3: Synthesize answer
        if stream:
            return self.synthesize_answer(question, retrieval_results, analysis)
        else:
            # Collect all chunks
            answer = ""
            for chunk in self.synthesize_answer(question, retrieval_results, analysis):
                answer += chunk
            return answer
    
    def get_reasoning_chain(self) -> List[Dict]:
        """Get the reasoning chain for transparency"""
        return self.reasoning_chain
    
    def explain_strategy(self, question: str) -> Dict:
        """
        Explain what strategy would be used for a question without executing it.
        Useful for understanding the agent's decision-making.
        """
        analysis = self.analyze_query(question)
        
        return {
            "question": question,
            "analysis": analysis,
            "explanation": f"""
Query Type: {analysis.get('query_type', 'unknown')}
Complexity: {analysis.get('complexity_score', 0)}/5
Primary Strategy: {analysis.get('primary_strategy', 'unknown')}
Reasoning: {analysis.get('reasoning', 'N/A')}

This strategy was chosen because: {analysis.get('reasoning', 'no reasoning provided')}
"""
        }
    
    def explore_entity(self, entity_identifier: str, depth: int = 2) -> Dict:
        """
        Explore an entity and its neighborhood in the graph.
        Useful for discovery and exploration queries.
        """
        self._add_reasoning_step(f"Exploring entity: {entity_identifier}")
        
        with self.driver.session() as session:
            query = f"""
            MATCH (n)
            WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $identifier)
            WITH n
            MATCH path = (n)-[*0..{depth}]-(connected)
            RETURN n as entity, 
                   labels(n) as entity_labels,
                   collect(DISTINCT connected) as connected_entities,
                   collect(DISTINCT relationships(path)) as relationships
            LIMIT 1
            """
            
            try:
                result = session.run(query, identifier=entity_identifier)
                record = result.single()
                
                if record:
                    return {
                        "entity": dict(record['entity']),
                        "labels": record['entity_labels'],
                        "connected_count": len(record['connected_entities']),
                        "relationship_types": list(set([
                            r.type for rels in record['relationships'] 
                            for r in rels if rels
                        ]))
                    }
            except Exception as e:
                logger.error(f"Entity exploration error: {e}")
        
        return {"error": "Entity not found"}
    
    def get_statistics(self) -> Dict:
        """Get graph statistics for overview"""
        with self.driver.session() as session:
            stats = {}
            
            # Node count by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(n) as count
                ORDER BY count DESC
            """)
            stats['node_counts'] = {r['type']: r['count'] for r in result}
            
            # Relationship count by type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            stats['relationship_counts'] = {r['type']: r['count'] for r in result}
            
            # Total counts
            result = session.run("MATCH (n) RETURN count(n) as total")
            stats['total_nodes'] = result.single()['total']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
            stats['total_relationships'] = result.single()['total']
            
            return stats