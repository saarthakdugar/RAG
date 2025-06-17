from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
import logging
from .. import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        # Check for required Azure OpenAI environment variables
        if not config.AZURE_OPENAI_API_KEY or not config.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required for Azure OpenAI embeddings. Please check your .env file.")

        # Initialize Azure OpenAI embeddings
        self.embeddings_model = AzureOpenAIEmbeddings(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME or "text-embedding-3-small",
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            dimensions=config.EMBEDDING_DIMENSIONS,
            timeout=60
        )
        self.vector_store = None

    def _documents_to_texts_and_metadatas(self, documents: List[Document]) -> tuple:
        """
        Extract texts for embedding from raw_content in metadata 
        instead of from page_content.
        """
        texts = []
        metadatas = []
        
        for doc in documents:
            # Use raw_content from metadata if available, otherwise use page_content
            if "raw_content" in doc.metadata:
                text_for_embedding = doc.metadata["raw_content"]
            else:
                text_for_embedding = doc.page_content
                # Add raw_content to metadata to ensure consistency
                doc.metadata["raw_content"] = doc.page_content
            
            texts.append(text_for_embedding)
            metadatas.append(doc.metadata)
            
        return texts, metadatas

    def build_vector_store_from_documents(self, documents):
        """Builds a FAISS vector store from a list of Langchain Document objects."""
        if not documents:
            logger.warning("No documents provided to build the vector store.")
            return
        try:
            logger.info(f"Building vector store with {len(documents)} document chunks using Azure OpenAI embeddings.")
            
            # Extract texts for embedding from raw_content in metadata
            texts, metadatas = self._documents_to_texts_and_metadatas(documents)
            
            # Build the vector store using the extracted texts and metadata
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings_model,
                metadatas=metadatas
            )
            logger.info("Vector store built successfully.")
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            self.vector_store = None # Ensure vector_store is None if building fails

    def add_documents_to_store(self, documents):
        """Adds new documents to an existing vector store."""
        if not self.vector_store:
            logger.info("Vector store not initialized. Building a new one.")
            self.build_vector_store_from_documents(documents)
            return
        if not documents:
            logger.warning("No documents provided to add to the vector store.")
            return
        try:
            logger.info(f"Adding {len(documents)} new document chunks to existing vector store.")
            
            # Extract texts for embedding from raw_content in metadata
            texts, metadatas = self._documents_to_texts_and_metadatas(documents)
            
            # Add to the vector store using the extracted texts and metadata
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            logger.info("Documents added successfully.")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")

    def search(self, query: str, k: int = 5) -> list:
        """Performs a similarity search in the vector store."""
        if not self.vector_store:
            logger.warning("Vector store is not initialized. Cannot perform search.")
            return []
        try:
            # Use RETRIEVED_CHUNK_LIMIT from config if available and not None/empty, otherwise use the provided k value
            chunk_limit = getattr(config, "RETRIEVED_CHUNK_LIMIT", None)
            if chunk_limit is None or chunk_limit <= 0:
                chunk_limit = k
            results = self.vector_store.similarity_search(query, k=chunk_limit)
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def save_vector_store(self, path: str = "faiss_index"):
        if self.vector_store:
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
        else:
            logger.warning("No vector store to save.")

    def load_vector_store(self, path: str = "faiss_index"):
        if not config.AZURE_OPENAI_API_KEY or not config.AZURE_OPENAI_ENDPOINT: # Check before attempting to load
            logger.error("Azure OpenAI credentials are not set. Cannot load vector store as embeddings model cannot be initialized.")
            self.vector_store = None
            return
        try:
            if not self.embeddings_model:
                self.embeddings_model = AzureOpenAIEmbeddings(
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                    azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME or "text-embedding-3-small",
                    api_key=config.AZURE_OPENAI_API_KEY,
                    api_version=config.AZURE_OPENAI_API_VERSION,
                    dimensions=config.EMBEDDING_DIMENSIONS,
                    timeout=60
                )

            self.vector_store = FAISS.load_local(path, self.embeddings_model, allow_dangerous_deserialization=True)
            logger.info(f"Vector store loaded from {path}")
        except FileNotFoundError:
            # Silent when vector store doesn't exist - this is expected on first run
            self.vector_store = None
        except Exception as e:
            # Changed from error to info as this is expected behavior on first run
            logger.info(f"No existing vector store found at {path}. A new one will be created when documents are processed.")
            self.vector_store = None 