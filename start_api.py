"""
Simple startup script for the FastAPI application
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        'NEO4J_URI',
        'NEO4J_USERNAME',
        'NEO4J_PASSWORD',
        'GROQ_API_KEY',
        'COHERE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error("❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        logger.error("\nPlease set these variables in your .env file")
        return False
    
    logger.info("✅ All required environment variables are set")
    return True


def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              RDBMS KNOWLEDGE GRAPH FASTAPI SERVER                       ║
║                                                                          ║
║  🚀 RESTful APIs for Ontology, Migration, and Q&A                       ║
║  🤖 Agentic AI for Intelligent Query Processing                         ║
║  📊 Interactive Documentation at /docs                                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main startup function"""
    print_banner()
    
    # Check environment
    if not check_environment():
        logger.error("\n❌ Environment check failed. Exiting.")
        sys.exit(1)
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"\n🌐 Starting server on http://{host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"📖 ReDoc: http://{host}:{port}/redoc")
    logger.info(f"❤️  Health Check: http://{host}:{port}/health\n")
    
    # Start server
    try:
        import uvicorn
        uvicorn.run(
            "api_main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    except ImportError:
        logger.error("❌ uvicorn not installed. Please run: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"\n❌ Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
