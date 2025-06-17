import asyncio
from ..core.llm_service import LLMService
from ..core.vector_store import VectorStoreManager
from ..processing.file_parser import AsyncFileParser
from ..prompt_manager.prompts import SYSTEM_PROMPT, Agent_Prompt
from collections import deque
import logging
from .. import config
from ..utils.history_manager import HistoryManager

logger = logging.getLogger(__name__)

class AliceAgent:
    def __init__(self, input_document_path: str = None, max_history_len: int = 5, vector_store_path: str = "faiss_index", session_id: str = None):
        logger.info(f"Initializing with input_path: {input_document_path}, vector_store: {vector_store_path}, session_id: {session_id}")
        try:
            self.llm_service = LLMService()
        except ValueError as e:
            logger.error(f"Failed to initialize LLMService: {e}")
            raise
        
        self.vector_store_manager = VectorStoreManager()
        self.file_parser = AsyncFileParser()
        self.chat_history_for_prompt = deque(maxlen=max_history_len)
        
        # Initialize history manager with session_id if provided
        self.persistent_history_manager = HistoryManager(session_id)
        
        # If this is an existing session with history, load it into the chat history for prompt
        if session_id and self.persistent_history_manager.session_started and self.persistent_history_manager.messages:
            # Load last max_history_len messages into the prompt history
            messages = self.persistent_history_manager.messages[-max_history_len:]
            for msg in messages:
                self.chat_history_for_prompt.append((msg['query'], msg['answer']))
            logger.info(f"Loaded {len(messages)} messages from existing session history into prompt context")
        
        # Use a common vector store path for all sessions (no session-specific postfix)
        self.vector_store_path = vector_store_path
        self.current_input_path = input_document_path

        # Attempt to load existing vector store, but don't worry if it's not found
        self.vector_store_manager.load_vector_store(self.vector_store_path)
        
        # Only log a message if we have input documents and need to process them
        if self.current_input_path and not self.vector_store_manager.vector_store:
            logger.info(f"Will process documents from {self.current_input_path}")
        # If no vector store and no input path, this is fine for chat-only use
        elif not self.current_input_path and not self.vector_store_manager.vector_store:
            logger.debug("No input path provided and no existing vector store found. Operating in chat-only mode.")
        # Only log success if we actually loaded a store
        elif self.vector_store_manager.vector_store:
            logger.info("Using existing vector store")

    async def initialize_documents(self):
        logger.info("Initializing documents")
        if self.current_input_path and not self.vector_store_manager.vector_store:
             # Always process documents on new run if path is provided
             logger.info(f"Processing documents from: {self.current_input_path}")
             await self.load_documents_from_path(self.current_input_path)
        elif self.vector_store_manager.vector_store and self.current_input_path:
            # If we have a vector store AND an input path, rebuild to ensure fresh data
            logger.info(f"Rebuilding vector store with documents from: {self.current_input_path}")
            await self.load_documents_from_path(self.current_input_path)
        elif self.vector_store_manager.vector_store:
            logger.info("Using existing vector store")
        elif not self.current_input_path:
            logger.debug("No initial path provided. Operating in chat-only mode.")

    async def load_documents_from_path(self, path: str):
        logger.info(f"Loading documents from: {path}")
        docs = await self.file_parser.process_path_intelligently(path)
        if docs:
            logger.info(f"Building vector store with {len(docs)} documents")
            self.vector_store_manager.build_vector_store_from_documents(docs)
            self.vector_store_manager.save_vector_store(self.vector_store_path)
            logger.info(f"Vector store built and saved successfully")
        else:
            logger.warning(f"No documents were processed from: {path}")
        self.current_input_path = path

    async def add_file_or_directory(self, path: str):
        logger.info(f"Adding file/directory: {path}")
        docs = await self.file_parser.process_path_intelligently(path)
        if docs:
            logger.info(f"Adding {len(docs)} documents to vector store")
            self.vector_store_manager.add_documents_to_store(docs)
            self.vector_store_manager.save_vector_store(self.vector_store_path)
            logger.info(f"Documents added to knowledge base successfully")
        else:
            logger.warning(f"No documents processed from: {path}")
            
    async def add_files_from_urls(self, urls: list):
        """
        Add files from the provided URLs to the vector store.
        """
        if not urls:
            logger.warning("No URLs provided for processing.")
            return

        valid_urls = [url for url in urls if url.startswith(("http://", "https://"))]
        if not valid_urls:
            logger.error("No valid URLs provided. Ensure URLs start with 'http://' or 'https://'.")
            return

        logger.info(f"Processing files from {len(valid_urls)} valid URLs.")
        try:
            docs = await self.file_parser.process_urls(valid_urls)
            if docs:
                logger.info(f"Adding {len(docs)} documents to vector store.")
                self.vector_store_manager.add_documents_to_store(docs)
                self.vector_store_manager.save_vector_store(self.vector_store_path)
                logger.info("Documents added to knowledge base successfully.")
            else:
                logger.warning("No documents processed from the provided URLs.")
        except Exception as e:
            logger.error(f"Error while processing URLs: {e}", exc_info=True)

    def _get_chat_history_for_prompt_str(self) -> str:
        history_str = ""
        for user_msg, ai_msg in self.chat_history_for_prompt:
            history_str += f"User: {user_msg}\nAI: {ai_msg}\n"
        return history_str.strip()

    def _format_context_from_docs(self, docs: list) -> str:
        """
        Format retrieved documents into a context string for the LLM,
        using the raw content from metadata instead of page_content.
        """
        if not docs:
            return "No relevant documents found."
            
        formatted_contexts = []
        
        for i, doc in enumerate(docs):
            meta = doc.metadata
            
            # Get content from metadata if available, otherwise use page_content
            content = meta.get("raw_content", doc.page_content).strip()
            
            # Simple document header with minimal metadata
            doc_header = f"[Document {i+1}] {meta.get('filename', '')}"
            
            # Add page number if available
            if "page" in meta:
                doc_header += f" (Page {meta['page']})"
            elif "chunk_index" in meta and "total_chunks" in meta and meta["total_chunks"] > 1:
                doc_header += f" (Section {meta['chunk_index'] + 1} of {meta['total_chunks']})"
            
            # Add minimal document information
            doc_meta = ""
            if "title" in meta:
                doc_meta += f"Title: {meta['title']}\n"
            if "author" in meta:
                doc_meta += f"Author: {meta['author']}\n"
            if "dates" in meta and meta["dates"]:
                doc_meta += f"Dates mentioned: {', '.join(meta['dates'])}\n"
            
            # Format the content with minimal context
            formatted_context = f"{doc_header}\n{doc_meta}\n{content}"
            formatted_contexts.append(formatted_context)
        
        # Combine all contexts with clear separation
        return "\n\n" + "\n\n---\n\n".join(formatted_contexts)

    async def ask(self, question: str) -> str:
        logger.info(f"Processing question: '{question[:50]}{'...' if len(question) > 50 else ''}'")
        
        self.persistent_history_manager.start_session_if_needed(question)
        chat_history_for_prompt_str = self._get_chat_history_for_prompt_str()
        query_for_retrieval = question 

        # Default context message if no documents or context found
        context_str = "No documents have been loaded or no relevant context found."
        context_docs = []
        if self.vector_store_manager.vector_store:
            logger.info(f"Searching vector store for relevant context")
            context_docs = self.vector_store_manager.search(query_for_retrieval, k=3)
            if context_docs:
                context_str = self._format_context_from_docs(context_docs)
                logger.info(f"Retrieved and formatted {len(context_docs)} relevant context snippets")
            else:
                 logger.info(f"No relevant context found in vector store")
        else:
            logger.warning("No vector store available for context retrieval")

        # Format the prompt with the retrieved context
        prompt = Agent_Prompt.format(
            system_prompt=SYSTEM_PROMPT,
            context=context_str,
            chat_history=chat_history_for_prompt_str,
            question=query_for_retrieval
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        logger.info("Sending request to LLM for generation")
        answer = await self.llm_service.generate_response(messages)
        logger.info(f"LLM response received ({len(answer)} chars)")
        
        # Update chat history
        self.chat_history_for_prompt.append((question, answer))
        
        # Use the version with source chunks if enabled in config
        if config.SOURCE_IN_HISTORY and context_docs:
            logger.info("SOURCE_IN_HISTORY is enabled, storing source chunks in history")
            self.persistent_history_manager.add_interaction_with_source(question, answer, context_docs)
        else:
            self.persistent_history_manager.add_interaction(question, answer)
        
        return answer

    def close(self):
        logger.info("Closing AliceAgent resources")
        if hasattr(self.file_parser, 'close') and callable(self.file_parser.close):
            self.file_parser.close()
        if self.vector_store_manager.vector_store:
             logger.info("Saving vector store before closing")
             self.vector_store_manager.save_vector_store(self.vector_store_path)
        logger.info(f"Chat history saved for session {self.persistent_history_manager.session_id}")
        logger.info("AliceAgent closed successfully") 