from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import logging
import asyncio
import uuid
from typing import List, Optional, Dict, Any
import datetime

from ..core.agent import AliceAgent
from .. import config
from ..utils.logger_config import setup_app_logging
from ..utils.history_manager import HistoryManager

# Configure logging
LOG_FILE_PATH = setup_app_logging(log_level=logging.INFO, console_log_level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent API", description="API for the AI Agent", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
active_agents: Dict[str, AliceAgent] = {}

# Initialize default RAG at startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing default RAG instance at startup...")
    try:
        default_session_id = "default"
        
        # Create a fresh agent that will always process documents if available
        agent = AliceAgent(
            input_document_path=config.INPUT_PATH,
            vector_store_path="faiss_index"
        )
        
        # This will now create a fresh vector store if input_path is provided
        await agent.initialize_documents()
        active_agents[default_session_id] = agent
        
        # Check if we have a vector store after initialization
        if agent.vector_store_manager.vector_store:
            logger.info("Default RAG instance initialized with document embeddings")
        else:
            logger.info("Default RAG instance initialized (no documents/embeddings available)")
    except Exception as e:
        logger.error(f"Error initializing default RAG instance: {e}", exc_info=True)
        # Continue server startup even if RAG initialization fails

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask Alice")
    session_id: Optional[str] = Field(None, description="Session ID (if continuing an existing session)")

class QueryResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for this conversation")
    response: str = Field(..., description="Alice's response to the query")
    timestamp: str = Field(..., description="Timestamp of the response")
    formatted_time: str = Field(..., description="Formatted time of the response")

class SessionResponse(BaseModel):
    session_id: str = Field(..., description="New session ID")
    message: str = Field(..., description="Status message")

class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]] = Field(..., description="List of active sessions")

class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]] = Field(..., description="Chat history for the session")
    metadata: Dict[str, Any] = Field(..., description="Session metadata")

# Simplified dependency to get agent for a session
async def get_agent(session_id: str):
    if session_id in active_agents:
        return active_agents[session_id]
    
    # Load agent from existing saved session
    try:
        # Check if history file exists
        history_manager = HistoryManager(session_id)
        if not history_manager.session_started:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
        # Try to use the default agent's vector store if available
        default_agent = active_agents.get("default")
        if default_agent:
            # Create a new agent for this session but use the same vector store
            agent = AliceAgent(
                vector_store_path="faiss_index",
                session_id=session_id
            )
            # Share the vector store from default agent
            agent.vector_store_manager = default_agent.vector_store_manager
            active_agents[session_id] = agent
            logger.info(f"Loaded existing session with shared RAG: {session_id}")
        else:
            # Fallback to creating a new agent with its own resources
            agent = AliceAgent(
                input_document_path=config.INPUT_PATH, 
                vector_store_path=f"faiss_index",
                session_id=session_id
            )
            await agent.initialize_documents()
            active_agents[session_id] = agent
            logger.info(f"Loaded existing session (no shared RAG): {session_id}")
        
        return agent
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or could not be loaded")

# Routes
@app.post("/query", response_model=QueryResponse)
async def ask_query(request: QueryRequest):
    session_id = request.session_id
    
    # Use default session if none provided
    if not session_id:
        if "default" in active_agents:
            session_id = str(uuid.uuid4())  # Create a new unique session ID
            logger.info(f"Creating new session with ID: {session_id}")
        else:
            # If default agent isn't available, use a generic ID
            session_id = str(uuid.uuid4())
            logger.info(f"No default agent available, creating new session: {session_id}")
    
    # Get or create the agent for this session
    agent = None
    
    # Check if session is already active in memory
    if session_id in active_agents:
        agent = active_agents[session_id]
        logger.info(f"Using existing active session: {session_id}")
    else:
        # Try to load existing session from disk
        try:
            history_manager = HistoryManager(session_id)
            if history_manager.session_started:
                # Create agent for existing history but use the default RAG
                default_agent = active_agents.get("default")
                if default_agent:
                    # Create a new agent that shares the same vector store as the default
                    agent = AliceAgent(
                        session_id=session_id,
                        vector_store_path="faiss_index"  # Use same vector store path as default
                    )
                    # Share the vector store from default agent
                    agent.vector_store_manager = default_agent.vector_store_manager
                    active_agents[session_id] = agent
                    logger.info(f"Loaded existing session from disk with shared RAG: {session_id}")
                else:
                    # Fallback if default agent isn't available
                    agent = AliceAgent(
                        input_document_path=config.INPUT_PATH,
                        vector_store_path="faiss_index",
                        session_id=session_id
                    )
                    await agent.initialize_documents()
                    active_agents[session_id] = agent
                    logger.info(f"Loaded existing session from disk (no shared RAG): {session_id}")
            else:
                # Create new agent with provided session ID, using default RAG
                default_agent = active_agents.get("default")
                if default_agent:
                    # Create a new agent that shares the same vector store as the default
                    agent = AliceAgent(
                        session_id=session_id,
                        vector_store_path="faiss_index"  # Use same vector store path as default
                    )
                    # Share the vector store from default agent
                    agent.vector_store_manager = default_agent.vector_store_manager
                    active_agents[session_id] = agent
                    logger.info(f"Created new session with shared RAG: {session_id}")
                else:
                    # Fallback if default agent isn't available
                    agent = AliceAgent(
                        input_document_path=config.INPUT_PATH,
                        vector_store_path="faiss_index",
                        session_id=session_id
                    )
                    await agent.initialize_documents()
                    active_agents[session_id] = agent
                    logger.info(f"Created new session without shared RAG: {session_id}")
        except Exception as e:
            logger.error(f"Error loading or creating session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error with session: {str(e)}")
    
    # Update title immediately if it's empty
    history_manager = agent.persistent_history_manager
    if not history_manager.metadata.get("title"):
        history_manager.metadata["title"] = request.query[:20]  # First 20 chars
        history_manager._save_history()
        logger.info(f"Updated title for session {session_id} to: '{history_manager.metadata['title']}'")
    
    # Process the query with the agent
    try:
        response = await agent.ask(request.query)
        
        # Get timestamp for response
        timestamp = datetime.datetime.now()
        formatted_time = timestamp.strftime("%I:%M %p")
        
        return {
            "session_id": session_id,
            "response": response,
            "timestamp": timestamp.isoformat(),
            "formatted_time": formatted_time
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    session_id = str(uuid.uuid4())
    try:
        # Use the default RAG instance instead of creating a new one
        default_agent = active_agents.get("default")
        if default_agent:
            # Create a new agent that shares the same vector store as the default
            agent = AliceAgent(
                session_id=session_id,
                vector_store_path="faiss_index"  # Use same vector store path as default
            )
            # Share the vector store from default agent
            agent.vector_store_manager = default_agent.vector_store_manager
            active_agents[session_id] = agent
            logger.info(f"Created new session with shared RAG: {session_id}")
        else:
            # Fallback if default agent isn't available
            agent = AliceAgent(
                input_document_path=config.INPUT_PATH, 
                vector_store_path="faiss_index",
                session_id=session_id
            )
            await agent.initialize_documents()
            active_agents[session_id] = agent
            logger.info(f"Created new session without shared RAG: {session_id}")
        
        # Initialize history file with empty title but with timestamp
        # Title will be updated with first query content
        history_manager = agent.persistent_history_manager
        history_manager.metadata["title"] = ""  # Empty title initially
        history_manager.metadata["date"] = datetime.datetime.now().isoformat()
        history_manager.session_started = True
        history_manager._save_history()
        logger.info(f"Initialized empty history file for session {session_id}")
        
        return {
            "session_id": session_id,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    # Get all sessions from disk
    disk_sessions = HistoryManager.list_all_sessions()
    
    # Combine with currently active sessions in memory
    active_session_ids = set(active_agents.keys())
    
    # Add any active sessions not already in disk_sessions
    for session_id, agent in active_agents.items():
        if not any(s["session_id"] == session_id for s in disk_sessions):
            history_manager = agent.persistent_history_manager
            disk_sessions.append({
                "session_id": session_id,
                "title": history_manager.metadata.get("title", "New Chat"),
                "date": history_manager.metadata.get("date", "Unknown"),
                "message_count": len(history_manager.messages)
            })
    
    # Format timestamps and mark active sessions
    for session in disk_sessions:
        session["active"] = session["session_id"] in active_session_ids
        # Format date to human-readable form if it exists
        if session["date"] and session["date"] != "Unknown":
            try:
                # Parse ISO format to datetime
                dt = datetime.datetime.fromisoformat(session["date"])
                # Format to 12-hour format with AM/PM
                session["formatted_date"] = dt.strftime("%b %d, %Y %I:%M %p")
            except (ValueError, TypeError):
                session["formatted_date"] = session["date"]
    
    return {"sessions": disk_sessions}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    # Check if session exists in memory
    agent = None
    if session_id in active_agents:
        agent = active_agents[session_id]
    
    # Check if session exists on disk
    history_manager = HistoryManager(session_id)
    file_exists = history_manager._session_exists(session_id)
    
    if not agent and not file_exists:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    try:
        # Close and remove from memory if active
        if agent:
            agent.close()
            del active_agents[session_id]
            logger.info(f"Closed and removed active agent for session {session_id}")
        
        # Delete history file if it exists
        if file_exists:
            history_file_path = history_manager.get_history_file_path()
            if os.path.exists(history_file_path):
                os.remove(history_file_path)
                logger.info(f"Deleted history file: {history_file_path}")
            
            # No longer need to delete vector store as it's shared across sessions
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_session_history(session_id: str):
    # Try to get from active agents first
    if session_id in active_agents:
        agent = active_agents[session_id]
        history_manager = agent.persistent_history_manager
    else:
        # Try to load from disk if not in memory
        history_manager = HistoryManager(session_id)
        if not history_manager.session_started:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Format timestamps in messages for readability
    formatted_messages = []
    for msg in history_manager.messages:
        formatted_msg = msg.copy()
        if "timestamp" in formatted_msg:
            try:
                dt = datetime.datetime.fromisoformat(formatted_msg["timestamp"])
                formatted_msg["formatted_time"] = dt.strftime("%I:%M %p")
            except (ValueError, TypeError):
                formatted_msg["formatted_time"] = formatted_msg.get("timestamp", "Unknown")
        formatted_messages.append(formatted_msg)
    
    # Format metadata date if present
    metadata = history_manager.metadata.copy()
    if "date" in metadata and metadata["date"]:
        try:
            dt = datetime.datetime.fromisoformat(metadata["date"])
            metadata["formatted_date"] = dt.strftime("%b %d, %Y %I:%M %p")
        except (ValueError, TypeError):
            metadata["formatted_date"] = metadata["date"]
    
    return {
        "history": formatted_messages,
        "metadata": metadata
    }

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API server shutting down, closing all agents")
    for session_id, agent in active_agents.items():
        try:
            agent.close()
            logger.info(f"Closed agent for session {session_id}")
        except Exception as e:
            logger.error(f"Error closing agent for session {session_id}: {e}", exc_info=True) 

from fastapi import UploadFile, File, Form

@app.post("/upload")
async def upload_data(files: Optional[List[UploadFile]] = None, folder_path: Optional[str] = Form(None)):
    """
    Endpoint to upload multiple files or specify a folder path to create a vector store.
    """
    try:
        # Check if files or folder path is provided
        if not files and not folder_path:
            raise HTTPException(status_code=400, detail="Either files or folder path must be provided.")

        input_paths = []

        # If files are uploaded, save them to the 'uploads' folder
        if files:
            upload_folder = config.INPUT_PATH
            os.makedirs(upload_folder, exist_ok=True)
            for file in files:
                temp_file_path = os.path.join(upload_folder, file.filename)
                with open(temp_file_path, "wb") as f:
                    f.write(await file.read())
                input_paths.append(temp_file_path)
                logger.info(f"Uploaded file saved to: {temp_file_path}")
        else:
            # Use the provided folder path
            input_paths.append(folder_path)
            if not os.path.exists(folder_path):
                raise HTTPException(status_code=400, detail="The specified folder path does not exist.")
            logger.info(f"Using folder path: {folder_path}")

        # Create a new agent to process the input
        agent = AliceAgent(
            input_document_path=config.INPUT_PATH,
            vector_store_path="faiss_index"
        )

        # Initialize documents and create the vector store
        for input_path in input_paths:
            await agent.add_file_or_directory(input_path)

        # Check if the vector store was created successfully
        if agent.vector_store_manager.vector_store:
            logger.info("Vector store created successfully.")
            return {"success": True, "message": "Vector store created successfully."}
        else:
            logger.warning("No documents were processed. Vector store not created.")
            return {"success": False, "message": "No documents were processed. Vector store not created."}

    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")