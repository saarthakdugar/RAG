import uvicorn
import logging
from .utils.logger_config import setup_app_logging
from . import config

# Configure logging
LOG_FILE_PATH = setup_app_logging(log_level=logging.INFO, console_log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
HOST = "127.0.0.1"
DEFAULT_PORT = 8000

def main():
    """Main entry point for AI Agent - API Server
    
    This is the only way to access AI Agent functionality.
    The terminal interface has been completely removed.
    """
    # Use port from config if available, otherwise use default
    port = int(config.API_PORT) if config.API_PORT else DEFAULT_PORT
    
    logger.info(f"Starting AI Agent API server on {HOST}:{port}")
    logger.info(f"API docs available at http://{HOST}:{port}/docs")
    
    # Get the directory name dynamically instead of hardcoding it
    import os
    import sys
    package_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    
    uvicorn.run(
        f"{package_name}.api.server:app",
        host=HOST,
        port=port
    )

if __name__ == "__main__":
    main() 