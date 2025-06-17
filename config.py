import os
from dotenv import load_dotenv

load_dotenv(override=True) # Load environment variables from .env file

# Azure OpenAI Configuration (for Chat Model)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Azure OpenAI Configuration (for Embeddings)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")

# Direct OpenAI Configuration (for Embeddings Model)
OPENAI_API_KEY_FOR_EMBEDDINGS = os.getenv("OPENAI_API_KEY_FOR_EMBEDDINGS")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Path to the input file or directory for documents (still from .env as it's path-dependent)
INPUT_PATH = os.getenv("INPUT_PATH")


# SharePoint configurations
SHAREPOINT_DOCUMENT_LIBRARY = "https://tfssharepoint.lfnet.se/sites/Rally/Driftdokumentation/Forms/AllItems.aspx"

# File processing configurations (defined directly)
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv", ".xls"]  # Focus only on these formats
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

DISABLE_FILE_PROCESSING_PARALLELISM = False 

# Embedding configurations
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small dimensions 

# Search configurations
RETRIEVED_CHUNK_LIMIT = 10  # Number of chunks to retrieve from vector store

# Server configurations
API_PORT = 1090  # default port (8000)

# Include source documents in chat history
SOURCE_IN_HISTORY = True  # Set to True to include retrieved source chunks in history