import sys
import os

if __name__ == "__main__":
    # Import the api_server module using a relative import
    try:
        from . import api_server
        api_server.main()
    except ImportError:
        # Handle the case when running as a script
        # Get the current directory name
        current_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        print(f"Please run the API server with: python -m {current_dir}.api_server")
        sys.exit(0) 