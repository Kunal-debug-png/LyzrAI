"""
Main orchestration file for RDBMS Graph Agentic Retrieval System.

This system combines:
1. RDBMS to Neo4j migration with semantic ontology (from test_sql.py)
2. Cohere embeddings for semantic search
3. Agentic retrieval with multiple strategies
4. Natural language Q&A interface

Prerequisites:
- Run test_sql.py first to migrate RDBMS to Neo4j and generate ontology
- Ensure ontology_output.json exists
"""

import os
import sys
import logging
from dotenv import load_dotenv
from archived.process_rdbms_graph import process_rdbms_graph_sync
from rdbms_graph_qa import AgenticRDBMSGraphQA

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Load from environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")





def check_prerequisites():
    """Check if prerequisites are met"""
    logger.info("\nChecking prerequisites...")
    
    # Check for ontology file
    if not os.path.exists("ontology_output.json"):
        logger.error("[ERROR] ontology_output.json not found!")
        logger.error("   Please run test_sql.py first to migrate your RDBMS to Neo4j")
        return False
    
    logger.info("[OK] Ontology file found")
    
    # Check API keys
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key":
        logger.error("[ERROR] GROQ_API_KEY not configured")
        return False
    
    if not COHERE_API_KEY or COHERE_API_KEY == "your_cohere_api_key":
        logger.error("[ERROR] COHERE_API_KEY not configured")
        return False
    
    logger.info("[OK] API keys configured")
    return True


def initialize_system(force_reprocess: bool = False):
    """
    Initialize the complete system:
    1. Process graph and create embeddings
    2. Initialize QA system
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: GRAPH PROCESSING & EMBEDDING CREATION")
    logger.info("=" * 80)
    
    # Step 1: Process RDBMS graph and create embeddings
    vector_store, graph_metadata, content_hash = process_rdbms_graph_sync(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        cohere_api_key=COHERE_API_KEY,
        ontology_path="ontology_output.json",
        force_reprocess=force_reprocess
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: INITIALIZING AGENTIC Q&A SYSTEM")
    logger.info("=" * 80)
    
    # Step 2: Initialize QA system
    qa_system = AgenticRDBMSGraphQA(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        groq_api_key=GROQ_API_KEY,
        vector_store=vector_store,
        graph_metadata=graph_metadata
    )
    
    logger.info("✅ System initialized successfully!")
    logger.info(f"📊 Content Hash: {content_hash}")
    
    return qa_system, graph_metadata


def show_system_info(qa_system, graph_metadata):
    """Display system information and statistics"""
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 80)
    
    # Ontology info
    ontology = graph_metadata.get('ontology', {})
    logger.info(f"\n📚 Domain: {ontology.get('domain', 'Unknown')}")
    
    # Graph statistics
    stats = qa_system.get_statistics()
    logger.info(f"\n📊 Graph Statistics:")
    logger.info(f"   Total Nodes: {stats.get('total_nodes', 0)}")
    logger.info(f"   Total Relationships: {stats.get('total_relationships', 0)}")
    
    logger.info(f"\n📋 Entity Types (Top 10):")
    for entity_type, count in list(stats.get('node_counts', {}).items())[:10]:
        logger.info(f"   • {entity_type}: {count}")
    
    logger.info(f"\n🔗 Relationship Types (Top 10):")
    for rel_type, count in list(stats.get('relationship_counts', {}).items())[:10]:
        logger.info(f"   • {rel_type}: {count}")
    
    logger.info("\n" + "=" * 80)


def demonstrate_strategies(qa_system):
    """Demonstrate different retrieval strategies"""
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY DEMONSTRATION")
    logger.info("=" * 80)
    
    demo_questions = [
        "What entities are in the database?",
        "How are the main entities connected?",
        "Find all records with status 'active'",
        "What is the relationship between customers and orders?"
    ]
    
    logger.info("\nShowing strategy selection for sample queries:\n")
    
    for question in demo_questions:
        strategy_info = qa_system.explain_strategy(question)
        logger.info(f"❓ Question: {question}")
        logger.info(f"🤖 Strategy: {strategy_info['analysis'].get('primary_strategy', 'unknown')}")
        logger.info(f"💭 Reasoning: {strategy_info['analysis'].get('reasoning', 'N/A')}")
        logger.info("-" * 80)


def interactive_mode(qa_system):
    """Interactive Q&A mode with streaming responses"""
    logger.info("\n" + "=" * 80)
    logger.info("INTERACTIVE Q&A MODE")
    logger.info("=" * 80)
    logger.info("\n💡 Ask questions in natural language!")
    logger.info("   The AI agent will automatically choose the best retrieval strategy.")
    logger.info("\n🎯 Commands:")
    logger.info("   • Type your question to get an answer")
    logger.info("   • 'explain: <question>' - Show strategy without executing")
    logger.info("   • 'explore: <entity>' - Explore an entity's neighborhood")
    logger.info("   • 'stats' - Show graph statistics")
    logger.info("   • 'exit' - Quit the system")
    logger.info("\n" + "=" * 80 + "\n")
    
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                logger.info("\n👋 Goodbye! Thank you for using the Agentic Retrieval System.")
                break
            
            elif user_input.lower() == 'stats':
                stats = qa_system.get_statistics()
                logger.info("\n📊 Current Graph Statistics:")
                logger.info(f"   Nodes: {stats.get('total_nodes', 0)}")
                logger.info(f"   Relationships: {stats.get('total_relationships', 0)}")
                logger.info(f"\n   Entity Types: {len(stats.get('node_counts', {}))}")
                logger.info(f"   Relationship Types: {len(stats.get('relationship_counts', {}))}")
                continue
            
            elif user_input.lower().startswith('explain:'):
                question = user_input[8:].strip()
                strategy_info = qa_system.explain_strategy(question)
                logger.info("\n🤖 Strategy Analysis:")
                logger.info(strategy_info['explanation'])
                continue
            
            elif user_input.lower().startswith('explore:'):
                entity = user_input[8:].strip()
                logger.info(f"\n🔍 Exploring entity: {entity}")
                info = qa_system.explore_entity(entity, depth=2)
                if 'error' in info:
                    logger.info(f"   ❌ {info['error']}")
                else:
                    logger.info(f"   Labels: {info['labels']}")
                    logger.info(f"   Connected Entities: {info['connected_count']}")
                    logger.info(f"   Relationship Types: {', '.join(info['relationship_types'])}")
                continue
            
            # Regular question - stream the answer
            logger.info("\n🤖 Agent Response:")
            logger.info("-" * 80)
            
            answer_chunks = []
            for chunk in qa_system.ask(user_input, stream=True):
                print(chunk, end='', flush=True)
                answer_chunks.append(chunk)
            
            print()  # New line after streaming
            logger.info("-" * 80)
            
            # Show reasoning chain
            logger.info("\n🧠 Reasoning Chain:")
            for i, step in enumerate(qa_system.get_reasoning_chain(), 1):
                logger.info(f"   {i}. {step['step']}")
            
        except KeyboardInterrupt:
            logger.info("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"\n❌ Error: {e}")
            logger.info("Please try again with a different question.")


def run_example_queries(qa_system):
    """Run example queries to demonstrate capabilities"""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE QUERIES")
    logger.info("=" * 80)
    
    example_queries = [
        "What are the main entity types in this database?",
        "Show me some example relationships between entities",
        "What patterns or structures exist in the data?"
    ]
    
    for i, question in enumerate(example_queries, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Example {i}: {question}")
        logger.info('='*80)
        logger.info("\n🤖 Answer:")
        logger.info("-" * 80)
        
        for chunk in qa_system.ask(question, stream=True):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 80)
        
        # Show reasoning
        logger.info("\n🧠 Reasoning Steps:")
        for step in qa_system.get_reasoning_chain():
            logger.info(f"   • {step['step']}")
        
        if i < len(example_queries):
            input("\n⏸️  Press Enter to continue to next example...")


def main():
    """Main entry point"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("\n❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    try:
        # Initialize system
        qa_system, graph_metadata = initialize_system(force_reprocess=False)
        
        # Show system information
        show_system_info(qa_system, graph_metadata)
        
        # Show strategy demonstration
        demonstrate_strategies(qa_system)
        
        # Ask user if they want to run examples
        logger.info("\n" + "=" * 80)
        user_choice = input("\n Would you like to see example queries? (y/n): ").strip().lower()
        
        if user_choice == 'y':
            run_example_queries(qa_system)
        
        # Interactive mode
        interactive_mode(qa_system)
        
    except KeyboardInterrupt:
        logger.info("\n\n👋 System interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'qa_system' in locals():
            qa_system.close()
        logger.info("\n✅ System shutdown complete.")


if __name__ == "__main__":
    main()