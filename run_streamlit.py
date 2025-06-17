#!/usr/bin/env python
"""
Run the Alice RAG Agent with Streamlit UI
"""
import os
import sys
import logging
import subprocess
import importlib

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the package name dynamically from the directory name
package_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

# Import config dynamically
config = importlib.import_module(f"{package_name}.config")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting AI Assistant with Streamlit UI...")
        
        # Check if streamlit is installed
        try:
            # Get the path to the streamlit executable in the same environment
            import streamlit
            streamlit_path = os.path.join(os.path.dirname(sys.executable), "streamlit")
            
            # Path to the streamlit app
            app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")
            
            # Run streamlit
            cmd = [streamlit_path, "run", app_path]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Execute streamlit
            process = subprocess.Popen(cmd)
            
            # Determine API port
            api_port = int(config.API_PORT) if config.API_PORT else 8000
            
            # Print helpful message
            logger.info("Streamlit UI is starting...")
            logger.info(f"Make sure the API server is running in another terminal with: python -m {package_name}.api_server")
            logger.info(f"The API server is configured to run on port {api_port}")
            logger.info("The Streamlit UI will be available at http://localhost:8501")
            
            # Wait for process to complete
            process.wait()
            
        except ImportError:
            logger.error("Streamlit is not installed. Please install it with: pip install streamlit")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping Streamlit UI...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting Streamlit UI: {e}", exc_info=True)
        sys.exit(1) 