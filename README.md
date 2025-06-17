# AI Assistant

A powerful Retrieval-Augmented Generation (RAG) system that enhances LLM responses with context from your documents.

## Overview

AI Assistant leverages Azure OpenAI services to provide accurate, context-aware responses to your questions. By processing and indexing your documents, the system creates a searchable vector database that enables the AI to reference specific information when answering queries.

## Key Benefits

- **Document-Grounded Responses**: Get accurate answers based on the content of your documents
- **Customizable Context Retrieval**: Configure how many relevant document chunks are used for each query
- **Multi-Format Support**: Process PDF, DOCX, TXT, and CSV files with robust encoding handling
- **Persistent Chat History**: Maintain conversation context across sessions


## Prerequisites

- Python 3.12 or higher
- Azure OpenAI API access

## Configuration

The system is configured through the `config.py` file. Key settings include:

```python
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Embedding Configuration
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small dimensions

# Document Processing
INPUT_PATH = os.getenv("INPUT_PATH")
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv"]
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Retrieval Settings
RETRIEVED_CHUNK_LIMIT = 10  # Number of chunks to retrieve from vector store

# Server configurations
API_PORT = 8000  # If None, default port (8000) will be used

# Include source documents in chat history
SOURCE_IN_HISTORY = True  # Include source documents in chat history
```

These settings can be customized directly in the config file or by setting environment variables.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Brainstorm2605/ai_hackathon.git
   cd ai-assistant
   ```

2. Create and activate a virtual environment:
   ```
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Set up Azure OpenAI credentials in a `.env` file:
   ```
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_API_VERSION=your_api_version
   
   #folder location
   INPUT_PATH=path/to/your/documents
   
   #Logging Configuration 
   LOG_LEVEL_CONSOLE="ERROR" # DEBUG, INFO, WARNING, ERROR, CRITICAL
   LOG_LEVEL_FILE="INFO"    # DEBUG, INFO, WARNING, ERROR, CRITICAL
   
   #Server Configuration
   API_PORT=8080            # Optional: Custom port for the API server (default: 8000)
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Streamlit App

The Streamlit interface provides a user-friendly chat experience:

```bash
python run_streamlit.py
```

This will start the Streamlit server and open the UI in your default browser. You can interact with the AI Assistant by:
1. Entering questions in the chat interface
2. Viewing responses with citations from your documents

## Running the API Server

For programmatic access or building custom interfaces, you can run:

### From any location (as a module):
```bash
python -m folder_name.api_server
```

The API server runs on the port specified in your configuration (default: 8000) and provides the following endpoints:

- `POST /query`: Submit questions to the AI Assistant
- `POST /process`: Process new documents
- `GET /sessions`: List available chat sessions

Note: The Streamlit UI connects to the API server, so both need to be running for the UI to function properly.
