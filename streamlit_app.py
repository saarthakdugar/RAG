import streamlit as st
import requests
import json
import logging
import os
import sys
from datetime import datetime
import uuid
import importlib

# Add the parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the package name dynamically from the directory name
package_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

# Import config dynamically
config = importlib.import_module(f"{package_name}.config")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get API port from config
api_port = int(config.API_PORT) if config.API_PORT else 8000

# API endpoints
API_BASE_URL = f"http://127.0.0.1:{api_port}"

# Set page config
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS
st.markdown("""
<style>
    /* Hide warning message */
    .stWarning {
        display: none !important;
    }
    
    /* Dark theme */
    body {
        background-color: #0E1117 !important;
        color: white !important;
    }
    
    .main {
        background-color: #0E1117 !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb, section[data-testid="stSidebar"] {
        background-color: #1A1C24 !important;
    }
    
    /* Main container width */
    .main .block-container {
        max-width: 1000px !important;
        padding-bottom: 100px !important;
        margin: 0 auto !important;
    }
    
    /* Remove all default Streamlit elements */
    .stApp > header {
        display: none !important;
    }
    
    .main .element-container {
        width: 100% !important;
    }
    
    /* Chat message container */
    .chat-message-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
        padding: 10px 20px;
    }
    
    /* Generic message styles */
    .message-bubble {
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 70%;
        word-wrap: break-word;
    }
    
    /* User message */
    .user-message {
        background-color: #1E1E1E;
        align-self: flex-end;
        margin-left: auto;
        margin-right: 0;
    }
    
    /* Assistant message */
    .assistant-message {
        background-color: #2C2C2C;
        align-self: flex-start;
        margin-left: 0;
        margin-right: auto;
    }
    
    /* Fix Streamlit chat elements */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0 !important;
        width: 100% !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    
    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        text-align: right !important;
    }
    
    [data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
        display: inline-block !important;
        background-color: #1E1E1E !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
        max-width: 70% !important;
        text-align: left !important;
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        text-align: left !important;
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] [data-testid="stChatMessageContent"] {
        display: inline-block !important;
        background-color: #2C2C2C !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
        max-width: 70% !important;
    }
    
    /* Thinking indicator */
    .thinking-indicator {
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 10px;
        background-color: #2C2C2C;
        border-radius: 10px;
        max-width: 200px;
    }
    
    .thinking-text {
        color: white;
        margin-left: 10px;
    }
    
    /* Make spinner visible and centered */
    .stSpinner {
        text-align: left !important;
        margin-left: 0 !important;
        display: inline-block !important;
    }
    
    /* Chat input */
    .chat-input-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 500px;
        max-width: 90%;
        background-color: #0E1117;
        padding: 10px;
        z-index: 999;
    }
    
    .stTextInput input {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #4A4A4A !important;
        border-radius: 5px !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        width: 100% !important;
    }
    
    /* Welcome message */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
        color: white;
        text-align: center;
    }
    
    .welcome-heading {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        font-size: 1.2rem;
        color: #B0B0B0;
    }
    
</style>
""", unsafe_allow_html=True)

# JavaScript for fixing layout issues
st.markdown("""
<script>
function fixLayout() {
    // Fix chat messages layout
    const messages = document.querySelectorAll('[data-testid="stChatMessage"]');
    messages.forEach(msg => {
        if (msg.getAttribute('data-testid').includes('user')) {
            msg.style.textAlign = 'right';
        } else {
            msg.style.textAlign = 'left';
        }
    });
}

// Run on page load and periodically
document.addEventListener('DOMContentLoaded', fixLayout);
setInterval(fixLayout, 1000);
</script>
""", unsafe_allow_html=True)

# Session state initialization
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'needs_rerun' not in st.session_state:
    st.session_state.needs_rerun = False
if 'thinking' not in st.session_state:
    st.session_state.thinking = False

def load_sessions():
    """Load all available chat sessions from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions")
        if response.status_code == 200:
            data = response.json()
            sessions = data.get("sessions", [])
            # Sort sessions by date (newest first), handling None values
            sessions.sort(
                key=lambda x: x.get("date", "") or "",  # Convert None to empty string for comparison
                reverse=True
            )
            return sessions
        else:
            logger.error(f"Failed to load sessions: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        return []

def create_session():
    """Create a new chat session"""
    try:
        response = requests.post(f"{API_BASE_URL}/sessions")
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            st.session_state.current_session_id = session_id
            st.session_state.messages = []
            st.session_state.needs_rerun = True
            logger.info(f"Created new session: {session_id}")
            return session_id
        else:
            logger.error(f"Failed to create session: {response.status_code}")
            st.error("Failed to create new session")
            return None
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        st.error(f"Error: {str(e)}")
        return None

def select_session(session_id):
    """Select an existing chat session and load its history"""
    if not session_id or session_id == st.session_state.current_session_id:
        return
    
    try:
        st.session_state.current_session_id = session_id
        st.session_state.messages = []
        
        # Get session history
        response = requests.get(f"{API_BASE_URL}/history/{session_id}")
        if response.status_code == 200:
            data = response.json()
            history = data.get("history", [])
            
            # Convert history to message format and add to session state
            for msg in history:
                st.session_state.messages.append({"role": "user", "content": msg.get("query", "")})
                st.session_state.messages.append({"role": "assistant", "content": msg.get("answer", "")})
            
            logger.info(f"Loaded session: {session_id}")
            st.session_state.needs_rerun = True
        else:
            logger.error(f"Failed to get history: {response.status_code}")
            st.error(f"Failed to load history for session {session_id}")
    except Exception as e:
        logger.error(f"Error selecting session: {e}")
        st.error(f"Error: {str(e)}")

def delete_session(session_id):
    """Delete a chat session"""
    if not session_id:
        return
    
    try:
        response = requests.delete(f"{API_BASE_URL}/sessions/{session_id}")
        if response.status_code == 200:
            # If we deleted the current session, reset it
            if session_id == st.session_state.current_session_id:
                st.session_state.current_session_id = None
                st.session_state.messages = []
            
            logger.info(f"Deleted session: {session_id}")
            st.session_state.needs_rerun = True
        else:
            logger.error(f"Failed to delete session: {response.status_code}")
            st.error(f"Failed to delete session")
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        st.error(f"Error: {str(e)}")

def upload_folder(folder_path):
    """Upload a folder path to the /upload API."""
    try:
        # Prepare the payload
        payload = {"folder_path": folder_path}

        # Make the POST request to the /upload endpoint
        response = requests.post(
            f"{API_BASE_URL}/upload",
            headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data=payload
        )

        # Handle the response
        if response.status_code == 200:
            st.success("Folder uploaded successfully!")
            logger.info(f"Folder uploaded successfully: {folder_path}")
            return response.json()
        else:
            st.error(f"Failed to upload folder: {response.status_code}")
            logger.error(f"Failed to upload folder: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading folder: {str(e)}")
        logger.error(f"Error uploading folder: {e}", exc_info=True)
        return None
    
def chat_with_alice(message):
    """Send a message to the AI and get a response"""
    if not message or not message.strip():
        return
    
    # Create a new session if none is active
    if not st.session_state.current_session_id:
        session_id = create_session()
        if not session_id:
            st.error("Failed to create a new session. Please try again.")
            return
    
    # Add user message to chat immediately
    st.session_state.messages.append({"role": "user", "content": message})
    
    # Mark as thinking but don't rerun yet
    st.session_state.thinking = True
    
    try:
        # Call the API
        payload = {
            "query": message,
            "session_id": st.session_state.current_session_id
        }
        
        # Make API call with a timeout
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            ai_response = data.get("response", "No response from AI Assistant")
            
            # Add AI response to chat
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        else:
            error_msg = f"Error: API returned status code {response.status_code}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            logger.error(f"API error: {response.status_code}")
    except requests.Timeout:
        error_msg = "The request timed out. The server might be busy or unavailable."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        logger.error("API request timed out")
    except Exception as e:
        error_msg = f"Error communicating with API: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        logger.error(f"Exception in chat: {e}")
    finally:
        # Always reset thinking state
        st.session_state.thinking = False
        st.session_state.needs_rerun = True

# Display thinking indicator outside of the function
thinking_container = st.container()

# Handle rerun if needed (to avoid warning message)
if st.session_state.needs_rerun:
    st.session_state.needs_rerun = False
    st.rerun()

# Sidebar for sessions
with st.sidebar:
    st.title("AI Assistant")
    
    # New chat button
    if st.button("+ New Chat", use_container_width=True, key="new_chat_btn"):
        create_session()
    
    st.markdown("---")
    uploaded_files = st.file_uploader("Choose files to upload", type=["txt", "pdf", "docx", "csv", "doc"], accept_multiple_files=True)

    # Button to trigger the upload
    if uploaded_files:
        if st.button("Upload Files"):
            try:
                # Prepare the files for upload
                files = [("files", (file.name, file.getvalue())) for file in uploaded_files]

                # Make the POST request to the /upload endpoint
                response = requests.post(f"{API_BASE_URL}/upload", files=files)

                # Handle the response
                if response.status_code == 200:
                    st.success("Files uploaded successfully!")
                    logger.info(f"Files uploaded successfully: {[file.name for file in uploaded_files]}")
                else:
                    st.error(f"Failed to upload files: {response.status_code}")
                    logger.error(f"Failed to upload files: {response.status_code}, Response: {response.text}")
            except Exception as e:
                st.error(f"Error uploading files: {str(e)}")
                logger.error(f"Error uploading files: {e}", exc_info=True)

    # Session list
    st.subheader("CHAT HISTORY")
    sessions = load_sessions()
    
    if sessions:
        for i, session in enumerate(sessions):
            session_id = session.get("session_id")
            # Ensure title is a string and use proper formatting
            title = str(session.get("title", "") or f"New Chat")
            if not title.strip():
                title = f"New Chat"
                
            # Add timestamp if available
            timestamp = session.get("date")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%b %d, %Y %I:%M %p")
                except:
                    time_str = ""
            else:
                time_str = "New"
                
            # Container for each session
            col1, col2 = st.columns([5, 1])
            
            # Session button
            if col1.button(
                title, 
                key=f"session_{i}_{session_id}", 
                help=session_id,
                use_container_width=True,
                on_click=select_session,
                kwargs={"session_id": session_id}
            ):
                pass
                
            # Small timestamp under button
            col1.caption(time_str)
            
            # Delete button
            if col2.button(
                "🗑️", 
                key=f"delete_{i}_{session_id}", 
                help="Delete this chat",
                on_click=delete_session,
                kwargs={"session_id": session_id}
            ):
                pass
                
            # Add a small space between sessions
            st.markdown("<div style='margin-bottom: 10px'></div>", unsafe_allow_html=True)
    else:
        st.info("No chat history available")

# Main chat area
main_container = st.container()

with main_container:
    # Welcome message when no session is selected
    if not st.session_state.current_session_id or not st.session_state.messages:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-heading">Hello, I'm AI Assistant! <div class="animated-emoji"></div> </div>
            <div class="welcome-text">Ask me questions about your documents and I'll help you find answers.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.write(message["content"])

# Show thinking indicator in its own container
with thinking_container:
    if st.session_state.thinking:
        col1, col2 = st.columns([1, 9])
        with col1:
            st.spinner("")
        with col2:
            st.markdown('<div style="color: white; margin-top: 10px;">AI Assistant is thinking...</div>', unsafe_allow_html=True)

# Chat input at the bottom
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
prompt = st.chat_input("What can I help with?", key="chat_input", disabled=st.session_state.thinking)
st.markdown('</div>', unsafe_allow_html=True)

if prompt and not st.session_state.thinking:
    chat_with_alice(prompt)
    st.rerun()  # Rerun after starting the chat process

if __name__ == "__main__":
    logger.info("Streamlit app started")