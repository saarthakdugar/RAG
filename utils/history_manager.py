import json
import os
import datetime
import uuid
import logging
import glob

# Get the package root directory dynamically based on current file location
# __file__ is the path to history_manager.py
# os.path.dirname(__file__) is the utils directory
# os.path.join(os.path.dirname(__file__), '..') goes up one level to the package root
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HISTORY_DIR_NAME = "chat_history"
HISTORY_DIR_PATH = os.path.join(PACKAGE_ROOT, HISTORY_DIR_NAME)

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, session_id=None):
        """
        Initialize a history manager.
        :param session_id: If provided, attempt to load an existing session. Otherwise, create a new one.
        """
        if not os.path.exists(HISTORY_DIR_PATH):
            try:
                os.makedirs(HISTORY_DIR_PATH)
                logger.info(f"Created chat history directory: {HISTORY_DIR_PATH}")
            except OSError as e:
                logger.error(f"Error creating chat history directory {HISTORY_DIR_PATH}: {e}. History may not be saved.", exc_info=True)
                # Potentially raise an error or disable history saving if directory creation fails
        
        if session_id and self._session_exists(session_id):
            self.session_id = session_id
            self.history_file_path = os.path.join(HISTORY_DIR_PATH, f"chat_{self.session_id}.json")
            self._load_history()
            logger.info(f"Loaded existing session {session_id} from {self.history_file_path}")
        else:
            self.session_id = session_id or str(uuid.uuid4())
            self.session_started = False
            self.history_file_path = os.path.join(HISTORY_DIR_PATH, f"chat_{self.session_id}.json")
            self.metadata = {
                "session_id": self.session_id,
                "title": None,
                "date": None, # Session start date
            }
            self.messages = []
            logger.info(f"Created new session with ID: {self.session_id}")

    def _session_exists(self, session_id):
        """Check if a session with the given ID exists"""
        return os.path.exists(os.path.join(HISTORY_DIR_PATH, f"chat_{session_id}.json"))

    def _load_history(self):
        """Loads chat history from the JSON file for this session"""
        try:
            with open(self.history_file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                self.metadata = history_data.get("metadata", {})
                
                # Ensure title is initialized properly
                if "title" not in self.metadata:
                    self.metadata["title"] = ""
                    
                self.messages = history_data.get("messages", [])
                self.session_started = True
                logger.debug(f"Loaded history for session {self.session_id} with {len(self.messages)} messages")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading chat history from {self.history_file_path}: {e}", exc_info=True)
            # Initialize with empty data if there's an error
            self.metadata = {"session_id": self.session_id, "title": "", "date": None}
            self.messages = []
            self.session_started = False

    def start_session_if_needed(self, first_query: str):
        """Starts a new session if one hasn't been started yet."""
        if not self.session_started:
            # Only set title if it's currently empty or None
            if not self.metadata["title"]:
                self.metadata["title"] = first_query[:20]  # Use first 20 chars of query as title
            
            # Always set/update the date when starting a session
            if not self.metadata["date"]:
                self.metadata["date"] = datetime.datetime.now().isoformat()
                
            self.session_started = True
            logger.info(f"Chat session {self.session_id} started. Title: '{self.metadata['title']}'. File: {self.history_file_path}")
            # Save initial metadata (empty messages array)
            self._save_history()

    def add_interaction(self, query: str, answer: str):
        """Adds a user query and AI answer to the history and saves it."""
        if not self.session_started:
            # This case should ideally be handled by calling start_session_if_needed first
            logger.warning("add_interaction called before session started. Starting session with current query.")
            self.start_session_if_needed(query)
        
        interaction = {
            "query": query,
            "answer": answer,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.messages.append(interaction)
        logger.debug(f"Added interaction to session {self.session_id}. Query: '{query[:50]}...'")
        self._save_history()
    
    def add_interaction_with_source(self, query: str, answer: str, source_chunks: list = None):
        """
        Adds a user query, AI answer, and source chunks to the history and saves it.
        This is used when SOURCE_IN_HISTORY is True to include retrieved documents.
        
        Args:
            query: The user's question
            answer: The AI's response
            source_chunks: List of source documents used for generating the answer
        """
        if not self.session_started:
            logger.warning("add_interaction_with_source called before session started. Starting session with current query.")
            self.start_session_if_needed(query)
        
        interaction = {
            "query": query,
            "answer": answer,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add source chunks if provided
        if source_chunks:
            # Process source chunks to extract relevant metadata
            sources = []
            for chunk in source_chunks:
                source_info = {
                    "content": chunk.page_content,
                    "metadata": {}
                }
                
                # Add key metadata fields from the chunk
                if hasattr(chunk, 'metadata'):
                    # Common metadata fields to include
                    for key in ['filename', 'page', 'chunk_index', 'total_chunks', 
                               'title', 'author', 'content_type']:
                        if key in chunk.metadata:
                            source_info['metadata'][key] = chunk.metadata[key]
                
                sources.append(source_info)
            
            interaction["sources"] = sources
            logger.debug(f"Added interaction with {len(sources)} source chunks to session {self.session_id}.")
        else:
            logger.debug(f"Added interaction without source chunks to session {self.session_id}.")
        
        self.messages.append(interaction)
        self._save_history()

    def _save_history(self):
        """Saves the current session history (metadata and messages) to its JSON file."""
        if not self.metadata["date"]:
            logger.warning(f"Attempted to save history for session {self.session_id} but session start date is not set. This indicates an issue.")
            # Avoid saving if session wasn't properly started, or start it now if appropriate
            # For now, we'll just log and not save if critical metadata is missing.
            return

        history_data = {
            "metadata": self.metadata,
            "messages": self.messages
        }
        try:
            with open(self.history_file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Chat history for session {self.session_id} saved to {self.history_file_path}")
        except IOError as e:
            logger.error(f"Error saving chat history to {self.history_file_path}: {e}", exc_info=True)
        except TypeError as e:
            logger.error(f"Error serializing history data to JSON for {self.history_file_path}: {e}", exc_info=True)

    def get_history_file_path(self) -> str:
        return self.history_file_path
    
    @staticmethod
    def list_all_sessions():
        """Get a list of all saved sessions from the chat_history directory"""
        session_files = glob.glob(os.path.join(HISTORY_DIR_PATH, "chat_*.json"))
        sessions = []
        
        for file_path in session_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    metadata = history_data.get("metadata", {})
                    # Extract session_id from filename as backup
                    filename = os.path.basename(file_path)
                    session_id = metadata.get("session_id") or filename.replace("chat_", "").replace(".json", "")
                    
                    # The title might be empty string or None
                    title = metadata.get("title", "")
                    if title is None:
                        title = ""
                    
                    sessions.append({
                        "session_id": session_id,
                        "title": title,  # Will be rendered as "New Chat" in frontend if empty
                        "date": metadata.get("date", "Unknown"),
                        "message_count": len(history_data.get("messages", []))
                    })
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Error reading session file {file_path}: {e}", exc_info=True)
        
        # Sort by date (newest first)
        # Handle special cases for unknown or empty dates - put them at the top
        sessions.sort(key=lambda x: (
            # First sort key: Put entries with no dates at the top
            0 if not x.get("date") or x.get("date") == "Unknown" else 1,
            # Second sort key: For entries with dates, sort by date descending
            "" if not x.get("date") or x.get("date") == "Unknown" else datetime.datetime.fromisoformat(x.get("date")).isoformat()), 
            reverse=True
        )
        return sessions

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG) # Basic config for direct test

    manager = HistoryManager()
    print(f"History will be saved to: {manager.get_history_file_path()}")

    manager.start_session_if_needed("What is the weather like today?")
    manager.add_interaction("What is the weather like today?", "It is sunny and warm.")
    manager.add_interaction("Any chance of rain later?", "A slight chance in the evening.")


    
    # Second manager for a new session
    manager2 = HistoryManager()

    manager2.add_interaction("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!")