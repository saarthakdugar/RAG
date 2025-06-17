import logging
import os
import datetime
import uuid
import sys

# Determine the project root directory dynamically (parent of the package directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_DIR_NAME = "app_logs"
LOG_DIR_PATH = os.path.join(PROJECT_ROOT, LOG_DIR_NAME)

# Store the generated log file path globally within this module if needed elsewhere
CURRENT_LOG_FILE_PATH = None

# Define which modules' logs to filter out to reduce noise
FILTERED_MODULES = [
    'httpcore', 
    'openai._base_client', 
    'httpx',
    'asyncio'
]

class ModuleFilter(logging.Filter):
    """Filter that excludes logs from specific modules"""
    def filter(self, record):
        # Return False to discard the record
        for module in FILTERED_MODULES:
            if record.name.startswith(module):
                return False
        return True

def setup_app_logging(log_level=logging.INFO, console_log_level=logging.WARNING) -> str:
    """
    Sets up application-wide logging with improved formatting and filtering.
    - Creates a unique log file for each run in LOG_DIR_PATH.
    - Configures the root logger to output to this file.
    - Filters out noisy logs from third-party libraries.
    - Uses clean, readable formatting.
    Returns the path to the created log file.
    """
    global CURRENT_LOG_FILE_PATH

    if not os.path.exists(LOG_DIR_PATH):
        try:
            os.makedirs(LOG_DIR_PATH)
        except OSError as e:
            print(f"Error creating log directory {LOG_DIR_PATH}: {e}. Logs may not be saved.", file=sys.stderr)
            return ""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    
    # Get package name dynamically from the directory structure
    package_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_file_name = f"{package_name}_run_{timestamp}_{run_id}.log"
    
    CURRENT_LOG_FILE_PATH = os.path.join(LOG_DIR_PATH, log_file_name)

    # --- Configure Root Logger --- 
    root_logger = logging.getLogger() 
    root_logger.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # --- File Handler (for application logs) ---
    try:
        file_handler = logging.FileHandler(CURRENT_LOG_FILE_PATH, mode='a', encoding='utf-8')
        # Cleaner, more readable formatter for file logs
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        file_handler.addFilter(ModuleFilter())  # Apply filter to exclude noisy modules
        root_logger.addHandler(file_handler)
    except IOError as e:
        print(f"Error setting up file handler for {CURRENT_LOG_FILE_PATH}: {e}. File logging may fail.", file=sys.stderr)

    # --- Console Handler (for user-relevant logs) ---
    if console_log_level is not None:
        console_handler = logging.StreamHandler(sys.stdout) 
        # Very simple formatter for console
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(console_log_level)
        console_handler.addFilter(ModuleFilter())  # Apply filter to exclude noisy modules
        root_logger.addHandler(console_handler)

    return CURRENT_LOG_FILE_PATH

if __name__ == '__main__':
    # Example of how to use it directly (for testing logger_config.py)
    log_file = setup_app_logging(log_level=logging.INFO, console_log_level=logging.INFO)
    
    if log_file:
        logging.debug("This is a debug message (should be in file only unless console_log_level=DEBUG).")
        logging.info("This is an info message.")
        logging.warning("This is a warning message.")
        logging.error("This is an error message.")
        logging.critical("This is a critical message.")

        # Example from another module
        another_module_logger = logging.getLogger("my_other_module")
        another_module_logger.info("Log message from another_module_logger.")
    else:
        print("Logging setup failed.") 