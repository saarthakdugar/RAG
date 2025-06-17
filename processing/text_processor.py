from langchain.text_splitter import RecursiveCharacterTextSplitter
from .. import config

class TextProcessor:
    def __init__(self, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False, # Keep it simple for now
        )

    def split_text(self, text: str):
        """Splits a long text into smaller chunks."""
        if not text or not isinstance(text, str):
            return []
        return self.text_splitter.split_text(text)

    def split_documents(self, documents):
        """Splits a list of Langchain Document objects."""
        if not documents:
            return []
        return self.text_splitter.split_documents(documents) 